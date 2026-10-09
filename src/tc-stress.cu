// tc-stress: sustained tensor-core correctness test with per-SM error attribution.
//
// Each warp builds deterministic small-integer A/B fragments, runs a chain of CHAIN identical
// tensor-core MMAs (so the exact answer is CHAIN * A*B), and checks every accumulator element
// against A*B computed on the CUDA cores with plain integer math. Integer / small-int inputs make
// every result exact, so ANY mismatch is a hardware (or clock/power) fault, never rounding.
//   int8: mma.sync.m16n8k32.row.col.s32.s8.s8.s32   (what llama.cpp MMQ uses on Ampere)
//   fp16: mma.sync.m16n8k16.row.col.f32.f16.f16.f32 (cuBLAS / FA style)
// Errors are counted per %smid. A layout mistake in this file would show up as errors on EVERY SM
// from the first launch, which is distinguishable from a real fault.
//
//   tc-stress <seconds> [int8|fp16|both]      (runs on CUDA device 0; pick the card with
//                                               CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>)
// Build: nvcc -O2 -arch=sm_80 -o tc-stress tc-stress.cu
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cuda_fp16.h>

constexpr int CHAIN = 64, WARPS = 8, MAX_SM = 256;

__device__ __forceinline__ uint32_t smid() { uint32_t r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16); }

// value generators: element (r,c) of A / (k,n) of B for a given seed
__device__ __forceinline__ int8_t a8(uint32_t s, int r, int c) { return (int8_t)((int)(mix(s ^ (r * 64 + c)) % 255) - 127); }
__device__ __forceinline__ int8_t b8(uint32_t s, int k, int n) { return (int8_t)((int)(mix(s * 3u ^ (0x10000 + k * 16 + n)) % 255) - 127); }
__device__ __forceinline__ int a16(uint32_t s, int r, int c) { return (int)(mix(s ^ (0x20000 + r * 32 + c)) % 17) - 8; }
__device__ __forceinline__ int b16(uint32_t s, int k, int n) { return (int)(mix(s * 5u ^ (0x30000 + k * 16 + n)) % 17) - 8; }

__device__ __forceinline__ uint32_t pack8(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    return (uint32_t)(uint8_t)x0 | ((uint32_t)(uint8_t)x1 << 8) | ((uint32_t)(uint8_t)x2 << 16) | ((uint32_t)(uint8_t)x3 << 24);
}
__device__ __forceinline__ uint32_t packh(int x0, int x1) {
    __half2 h = __halves2half2(__int2half_rn(x0), __int2half_rn(x1)); return *(uint32_t *)&h;
}

__global__ void tc_kernel(uint32_t seed, int mode, unsigned long long *err_by_sm, unsigned long long *checks_by_sm,
                          unsigned long long *first_err) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;              // groupID, threadID_in_group
    const uint32_t s = mix(seed ^ (blockIdx.x * WARPS + warp) * 0x9e3779b9u);
    const uint32_t sm = smid() & (MAX_SM - 1);
    unsigned long long errs = 0, checks = 0;

    if (mode & 1) {   // ---- int8 m16n8k32 ----
        // A frag: a0 (row g, k t*4..+3), a1 (row g+8, same k), a2 (row g, k 16+t*4..), a3 (row g+8, k 16+..)
        uint32_t a[4], b[2];
        a[0] = pack8(a8(s, g, t*4), a8(s, g, t*4+1), a8(s, g, t*4+2), a8(s, g, t*4+3));
        a[1] = pack8(a8(s, g+8, t*4), a8(s, g+8, t*4+1), a8(s, g+8, t*4+2), a8(s, g+8, t*4+3));
        a[2] = pack8(a8(s, g, 16+t*4), a8(s, g, 16+t*4+1), a8(s, g, 16+t*4+2), a8(s, g, 16+t*4+3));
        a[3] = pack8(a8(s, g+8, 16+t*4), a8(s, g+8, 16+t*4+1), a8(s, g+8, 16+t*4+2), a8(s, g+8, 16+t*4+3));
        // B frag: b0 (k t*4..+3, col g), b1 (k 16+t*4.., col g)
        b[0] = pack8(b8(s, t*4, g), b8(s, t*4+1, g), b8(s, t*4+2, g), b8(s, t*4+3, g));
        b[1] = pack8(b8(s, 16+t*4, g), b8(s, 16+t*4+1, g), b8(s, 16+t*4+2, g), b8(s, 16+t*4+3, g));
        int c[4] = {0, 0, 0, 0};
        #pragma unroll 1
        for (int i = 0; i < CHAIN; i++)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        // C frag: c0,c1 (row g, col t*2+{0,1}); c2,c3 (row g+8, col t*2+{0,1})
        for (int i = 0; i < 4; i++) {
            const int row = g + (i >= 2 ? 8 : 0), col = t * 2 + (i & 1);
            int ref = 0;
            for (int k = 0; k < 32; k++) ref += (int)a8(s, row, k) * (int)b8(s, k, col);
            checks++;
            if (c[i] != ref * CHAIN) {
                errs++;
                atomicCAS(first_err, 0ULL, (1ULL << 63) | ((unsigned long long)sm << 40) | ((unsigned long long)(uint32_t)(c[i] - ref * CHAIN) & 0xffffffffULL));
            }
        }
    }
    if (mode & 2) {   // ---- fp16 m16n8k16, f32 accumulate ----
        // A frag (row-major 16x16): a0 (row g, k t*2..+1), a1 (row g+8, k t*2..), a2 (row g, k 8+t*2..), a3 (row g+8, k 8+t*2..)
        uint32_t a[4], b[2];
        a[0] = packh(a16(s, g, t*2), a16(s, g, t*2+1));
        a[1] = packh(a16(s, g+8, t*2), a16(s, g+8, t*2+1));
        a[2] = packh(a16(s, g, 8+t*2), a16(s, g, 8+t*2+1));
        a[3] = packh(a16(s, g+8, 8+t*2), a16(s, g+8, 8+t*2+1));
        // B frag (col-major 16x8): b0 (k t*2..+1, col g), b1 (k 8+t*2.., col g)
        b[0] = packh(b16(s, t*2, g), b16(s, t*2+1, g));
        b[1] = packh(b16(s, 8+t*2, g), b16(s, 8+t*2+1, g));
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 1
        for (int i = 0; i < CHAIN; i++)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        for (int i = 0; i < 4; i++) {
            const int row = g + (i >= 2 ? 8 : 0), col = t * 2 + (i & 1);
            int ref = 0;
            for (int k = 0; k < 16; k++) ref += a16(s, row, k) * b16(s, k, col);
            checks++;
            if (c[i] != (float)(ref * CHAIN)) {
                errs++;
                atomicCAS(first_err, 0ULL, (1ULL << 63) | (1ULL << 62) | ((unsigned long long)sm << 40) | ((unsigned long long)(uint32_t)((int)c[i] - ref * CHAIN) & 0xffffffffULL));
            }
        }
    }
    // warp-reduce, then one atomic per warp
    for (int o = 16; o; o >>= 1) { errs += __shfl_xor_sync(~0u, errs, o); checks += __shfl_xor_sync(~0u, checks, o); }
    if (lane == 0) { if (errs) atomicAdd(&err_by_sm[sm], errs); atomicAdd(&checks_by_sm[sm], checks); }
}

int main(int argc, char **argv) {
    const double seconds = argc > 1 ? atof(argv[1]) : 30;
    const char *m = argc > 2 ? argv[2] : "both";
    const int mode = !strcmp(m, "int8") ? 1 : !strcmp(m, "fp16") ? 2 : 3;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    char bus[32]; cudaDeviceGetPCIBusId(bus, sizeof bus, 0);
    printf("%s at %s: %d SMs, mode %s, %.0f s\n", p.name, bus, p.multiProcessorCount, m, seconds);
    unsigned long long *d_err, *d_chk, *d_first;
    cudaMalloc(&d_err, MAX_SM * 8); cudaMalloc(&d_chk, MAX_SM * 8); cudaMalloc(&d_first, 8);
    cudaMemset(d_err, 0, MAX_SM * 8); cudaMemset(d_chk, 0, MAX_SM * 8); cudaMemset(d_first, 0, 8);
    const int blocks = p.multiProcessorCount * 16;
    auto t0 = std::chrono::steady_clock::now(); double el = 0; long launches = 0; double next_report = 10;
    std::vector<unsigned long long> err(MAX_SM), chk(MAX_SM);
    while (el < seconds) {
        for (int i = 0; i < 20; i++) tc_kernel<<<blocks, WARPS * 32>>>((uint32_t)(launches * 20 + i), mode, d_err, d_chk, d_first);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 3; }
        launches++;
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= next_report || el >= seconds) {
            cudaMemcpy(err.data(), d_err, MAX_SM * 8, cudaMemcpyDeviceToHost);
            unsigned long long te = 0, tcnt = 0; for (int i = 0; i < MAX_SM; i++) { te += err[i]; }
            cudaMemcpy(chk.data(), d_chk, MAX_SM * 8, cudaMemcpyDeviceToHost); for (int i = 0; i < MAX_SM; i++) tcnt += chk[i];
            printf("  t=%4.0fs  checks %llu  errors %llu\n", el, tcnt, te); fflush(stdout);
            next_report += 10;
        }
    }
    cudaMemcpy(err.data(), d_err, MAX_SM * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(chk.data(), d_chk, MAX_SM * 8, cudaMemcpyDeviceToHost);
    unsigned long long first; cudaMemcpy(&first, d_first, 8, cudaMemcpyDeviceToHost);
    int sms_seen = 0, sms_bad = 0; unsigned long long te = 0;
    for (int i = 0; i < MAX_SM; i++) { if (chk[i]) sms_seen++; if (err[i]) { sms_bad++; te += err[i]; } }
    printf("SMs exercised: %d, SMs with errors: %d, total errors: %llu\n", sms_seen, sms_bad, te);
    if (sms_bad) {
        printf("errors by smid:"); for (int i = 0; i < MAX_SM; i++) if (err[i]) printf(" %d:%llu/%llu", i, err[i], chk[i]); printf("\n");
        printf("first error: %s on smid %llu, delta %d\n", (first >> 62 & 1) ? "fp16" : "int8", (first >> 40) & 0xff, (int)(first & 0xffffffff));
    }
    printf("RESULT: %s\n", te ? "FAIL" : (sms_seen < p.multiProcessorCount ? "PASS (but not every SM was exercised)" : "PASS"));
    return te ? 1 : 0;
}
