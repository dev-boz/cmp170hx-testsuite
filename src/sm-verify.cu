// sm-verify: per-SM correctness check for SM/TPC unlocks.
//
// Every block computes a deterministic function of its blockIdx only (tensor-core
// WMMA fp16->fp32, a long FP32 FMA chain, integer mixing, and a shared-memory
// pattern), so the per-block checksums must be bit-identical regardless of which
// SM runs the block or how many SMs the GPU has. Each block also records %smid.
//
//   sm-verify record <file>   run and save per-block checksums (baseline)
//   sm-verify check  <file>   run and compare against a saved baseline
//
// Build: nvcc -O2 -arch=sm_80 -o sm-verify sm-verify.cu
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

constexpr int BLOCKS = 16384, THREADS = 256, FMA_ITERS = 20000, REPS = 5;

__device__ __forceinline__ uint32_t smid() { uint32_t r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16); }

__global__ void work(uint64_t *out, uint32_t *sm) {
    __shared__ half a[16 * 16], b[16 * 16];
    __shared__ float c[8][16 * 16];
    __shared__ uint32_t pat[4096];
    const uint32_t bid = blockIdx.x, tid = threadIdx.x, warp = tid / 32;

    // shared-memory pattern: write, sync, read back a different thread's words
    for (int i = tid; i < 4096; i += THREADS) pat[i] = mix(bid * 4096u + i);
    for (int i = tid; i < 256; i += THREADS) {
        a[i] = __float2half(((mix(bid * 977u + i) & 1023) - 512) / 256.0f);
        b[i] = __float2half(((mix(bid * 131u + i) & 1023) - 512) / 256.0f);
    }
    __syncthreads();
    uint32_t sacc = 0;
    for (int i = (tid * 7) % 4096, k = 0; k < 16; k++, i = (i + 997) % 4096) sacc += pat[i] ^ (uint32_t)i;

    // tensor cores: each warp does 64 chained 16x16x16 MMAs
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::load_matrix_sync(fa, a, 16); wmma::load_matrix_sync(fb, b, 16);
    wmma::fill_fragment(fc, 0.0f);
    for (int k = 0; k < 64; k++) wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c[warp], fc, 16, wmma::mem_row_major);

    // FP32 FMA chain (fixed order per thread -> deterministic)
    float x = (mix(bid * THREADS + tid) & 0xffff) / 65536.0f, y = 0.999f;
    for (int i = 0; i < FMA_ITERS; i++) { x = fmaf(x, y, 0.0001f); y = fmaf(y, 0.9999f, 0.00001f); }

    __syncthreads();
    uint32_t tc = 0;
    for (int i = tid; i < 256; i += THREADS) tc ^= __float_as_uint(c[warp][i]) * (i + 1);
    uint64_t v = ((uint64_t)(sacc ^ tc) << 32) ^ __float_as_uint(x) ^ ((uint64_t)__float_as_uint(y) << 7);

    // reduce thread values in a fixed order through shared memory
    __shared__ uint64_t red[THREADS];
    red[tid] = mix((uint32_t)v) ^ ((uint64_t)mix((uint32_t)(v >> 32)) << 32) ^ tid;
    __syncthreads();
    if (tid == 0) {
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < THREADS; i++) h = (h ^ red[i]) * 1099511628211ULL;
        out[bid] = h; sm[bid] = smid();
    }
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "record") && strcmp(argv[1], "check"))) {
        fprintf(stderr, "usage: sm-verify record|check <file>\n"); return 2;
    }
    int dev = 0; cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("%s: %d SMs (multiProcessorCount)\n", p.name, p.multiProcessorCount);
    uint64_t *d_out; uint32_t *d_sm; cudaMalloc(&d_out, BLOCKS * 8); cudaMalloc(&d_sm, BLOCKS * 4);
    std::vector<uint64_t> out(BLOCKS), first(BLOCKS); std::vector<uint32_t> sm(BLOCKS);
    std::vector<int> bad_by_sm(256, 0), seen_sm(256, 0);
    int unstable = 0;
    for (int r = 0; r < REPS; r++) {
        work<<<BLOCKS, THREADS>>>(d_out, d_sm);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
        cudaMemcpy(out.data(), d_out, BLOCKS * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(sm.data(), d_sm, BLOCKS * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < BLOCKS; i++) {
            seen_sm[sm[i] & 255]++;
            if (r == 0) first[i] = out[i];
            else if (out[i] != first[i]) { unstable++; bad_by_sm[sm[i] & 255]++; }
        }
    }
    int nsm = 0; for (int s = 0; s < 256; s++) nsm += seen_sm[s] > 0;
    printf("blocks ran on %d distinct SM ids; %d run-to-run mismatches over %d reps\n", nsm, unstable, REPS);

    if (!strcmp(argv[1], "record")) {
        FILE *f = fopen(argv[2], "wb"); fwrite(first.data(), 8, BLOCKS, f); fclose(f);
        printf("baseline recorded to %s\n", argv[2]);
        return unstable ? 1 : 0;
    }
    std::vector<uint64_t> base(BLOCKS);
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(base.data(), 8, BLOCKS, f) != BLOCKS) { fprintf(stderr, "cannot read baseline\n"); return 2; }
    fclose(f);
    // re-run once more to attribute baseline mismatches to SMs
    work<<<BLOCKS, THREADS>>>(d_out, d_sm); cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, BLOCKS * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(sm.data(), d_sm, BLOCKS * 4, cudaMemcpyDeviceToHost);
    std::vector<int> wrong(256, 0), ran(256, 0); int total_wrong = 0;
    for (int i = 0; i < BLOCKS; i++) { ran[sm[i] & 255]++; if (out[i] != base[i]) { wrong[sm[i] & 255]++; total_wrong++; } }
    printf("vs baseline: %d / %d blocks differ\n", total_wrong, BLOCKS);
    for (int s = 0; s < 256; s++)
        if (wrong[s] || bad_by_sm[s]) printf("  SM %3d: %d/%d blocks wrong vs baseline, %d run-to-run mismatches\n", s, wrong[s], ran[s], bad_by_sm[s]);
    printf("RESULT: %s\n", (total_wrong || unstable) ? "FAIL - miscomputation detected" : "PASS - bit-identical on every SM");
    return (total_wrong || unstable) ? 1 : 0;
}
