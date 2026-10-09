// vram-memtest: device-side HBM test at full bandwidth, with per-address error reports.
// Fills (nearly) all free VRAM in 1 GiB chunks and cycles patterns until the time is up:
//   hash   - unique pseudo-random word per address (catches address/aliasing faults)
//   ~hash  - its bitwise inverse (every cell sees both 0 and 1)
//   check  - 0x55555555/0xAAAAAAAA alternating by word, then inverted
//   walk   - walking one: bit (addr + round) % 32 set
// Every 4th round waits `hold` seconds between write and verify (retention).
// Errors are counted per chunk; the first 32 are logged with address, value, expected and XOR.
//
//   vram-memtest [minutes=10] [hold_s=30] [reserve_MiB=1024] [pattern_seq=01234]   (device 0: CUDA_VISIBLE_DEVICES)
//   pattern_seq: digits 0-4 = hash ~hash check ~check walk, cycled in order (e.g. "40" alternates walk->hash)
// Build: nvcc -O2 -arch=sm_80 -o vram-memtest vram-memtest.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(err_), __LINE__); exit(3); } } while (0)

constexpr size_t CHUNK = 1ull << 30, WORDS = CHUNK / 4;
constexpr int MAXLOG = 32;
struct errlog { unsigned long long n; unsigned int logged; unsigned int or_xor;
                struct { unsigned int chunk; unsigned long long word; unsigned int got, want; } rec[MAXLOG]; };

__device__ __forceinline__ uint32_t expect(int pat, uint32_t seed, uint32_t chunk, uint64_t w) {
    switch (pat) {
    case 0: case 1: {
        uint64_t x = (w + 1 + ((uint64_t)chunk << 28)) * 0x9E3779B97F4A7C15ull ^ ((uint64_t)seed << 32);
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        return pat == 0 ? (uint32_t)x : ~(uint32_t)x; }
    case 2: return (w & 1) ? 0xAAAAAAAAu : 0x55555555u;
    case 3: return (w & 1) ? 0x55555555u : 0xAAAAAAAAu;
    default: return 1u << ((w + chunk + seed) & 31);
    }
}
__global__ void fill(uint32_t *d, int pat, uint32_t seed, uint32_t chunk) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < WORDS; i += (uint64_t)gridDim.x * blockDim.x)
        d[i] = expect(pat, seed, chunk, i);
}
__global__ void check(const uint32_t *d, int pat, uint32_t seed, uint32_t chunk, errlog *e, unsigned long long *per_chunk) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < WORDS; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t want = expect(pat, seed, chunk, i), got = d[i];
        if (got != want) {
            atomicAdd(&e->n, 1ull); atomicAdd(&per_chunk[chunk], 1ull); atomicOr(&e->or_xor, got ^ want);
            unsigned int k = atomicAdd(&e->logged, 1u);
            if (k < MAXLOG) { e->rec[k].chunk = chunk; e->rec[k].word = i; e->rec[k].got = got; e->rec[k].want = want; }
        }
    }
}
int main(int argc, char **argv) {
    const double minutes = argc > 1 ? atof(argv[1]) : 10;
    const int hold = argc > 2 ? atoi(argv[2]) : 30;
    const size_t reserve = (argc > 3 ? atoll(argv[3]) : 1024) << 20;
    const char *seq = argc > 4 ? argv[4] : "01234"; const int nseq = (int)strlen(seq);
    char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    std::vector<uint32_t *> ch;
    while (fr > reserve + CHUNK) { uint32_t *p; if (cudaMalloc(&p, CHUNK) != cudaSuccess) { cudaGetLastError(); break; } ch.push_back(p); CK(cudaMemGetInfo(&fr, &tot)); }
    const int nc = (int)ch.size();
    printf("device %s: testing %d GiB of %.1f GiB for %.0f min (hold %d s every 4th round), pattern sequence %s\n", bus, nc, tot / 1073741824.0, minutes, hold, seq);
    errlog *e; unsigned long long *pc; CK(cudaMallocManaged(&e, sizeof(errlog))); CK(cudaMallocManaged(&pc, nc * 8));
    for (int c = 0; c < nc; c++) pc[c] = 0;
    const char *names[] = {"hash", "~hash", "check", "~check", "walk"};
    auto t0 = std::chrono::steady_clock::now(); unsigned long long total = 0; double gbytes = 0; int round = 0;
    auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    while (el() < minutes * 60) {
        const int pat = seq[round % nseq] - '0'; const uint32_t seed = 0x1234567u * (round + 1);
        for (int c = 0; c < nc; c++) fill<<<4096, 256>>>(ch[c], pat, seed, c);
        CK(cudaDeviceSynchronize());
        if (round % 4 == 3 && hold > 0) sleep(hold);
        *e = {};
        for (int c = 0; c < nc; c++) check<<<4096, 256>>>(ch[c], pat, seed, c, e, pc);
        CK(cudaDeviceSynchronize());
        gbytes += 2.0 * nc * CHUNK / 1e9; total += e->n;
        if (e->n || round % 50 == 0) printf("  round %3d %-6s%s t=%5.0fs  errors %llu%s\n", round, names[pat], (round % 4 == 3 && hold > 0) ? " +hold" : "      ",
               el(), e->n, e->n ? "  <<<" : ""); fflush(stdout);
        for (unsigned k = 0; k < e->logged && k < MAXLOG && k < 8; k++)
            printf("      chunk %2u word %10llu (byte 0x%llx)  got %08x want %08x xor %08x\n", e->rec[k].chunk, e->rec[k].word,
                   e->rec[k].word * 4 + (unsigned long long)e->rec[k].chunk * CHUNK, e->rec[k].got, e->rec[k].want, e->rec[k].got ^ e->rec[k].want);
        round++;
    }
    printf("rounds %d, %.0f GB written+verified, total errors %llu\n", round, gbytes, total);
    if (total) { printf("errors by chunk:"); for (int c = 0; c < nc; c++) if (pc[c]) printf(" %d:%llu", c, pc[c]); printf("\n"); }
    printf("RESULT: %s\n", total ? "HBM ERRORS" : "PASS");
    return total ? 1 : 0;
}
