// hbm-map: are the HBM errors at FIXED physical locations (quarantinable) or wandering?
//
// Allocates the free VRAM as fixed physical blocks (CUDA VMM, cuMemCreate at the
// allocation granularity, 2 MiB on the 170HX), maps them into one arena, and keeps
// that SAME physical memory for the whole run. Fill/check rounds use the same
// patterns and launch shape as vram-memtest (1 GiB per launch, 4096x256), so seq
// "40" (walk->hash) reproduces the step-test stress. Unlike vram-memtest it logs
// EVERY bad word (up to CAP per round), grouped into 128-byte blocks, with the
// physical block (handle) it lives in.
//
//   hbm-map [minutes=10] [reserve_MiB=1024] [pattern_seq=40] [out_prefix=hbm-map]
//
// Writes <out_prefix>-blocks.csv (one line per bad 128 B block per round, flushed
// every round so a GSP crash loses nothing) and <out_prefix>-handles.csv (per-handle
// totals) plus a summary on stdout. Handle numbers are only stable WITHIN one run.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(err_), __LINE__); fflush(stdout); exit(3); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_; cuGetErrorString(r_, &s_); printf("CU error %s at line %d\n", s_, __LINE__); fflush(stdout); exit(3); } } while (0)

constexpr size_t REGION = 1ull << 30;           // bytes per kernel launch, as vram-memtest
constexpr unsigned CAP = 1u << 20;              // logged bad words per round
struct rec { unsigned long long w; unsigned got, want; };

__device__ __forceinline__ uint32_t expect(int pat, uint32_t seed, uint64_t w) {
    switch (pat) {
    case 0: case 1: {
        uint64_t x = (w + 1) * 0x9E3779B97F4A7C15ull ^ ((uint64_t)seed << 32);
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        return pat == 0 ? (uint32_t)x : ~(uint32_t)x; }
    case 2: return (w & 1) ? 0xAAAAAAAAu : 0x55555555u;
    case 3: return (w & 1) ? 0x55555555u : 0xAAAAAAAAu;
    default: return 1u << ((w + seed) & 31);
    }
}
__global__ void fill(uint32_t *d, uint64_t w0, uint64_t nw, int pat, uint32_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < nw; i += (uint64_t)gridDim.x * blockDim.x)
        d[i] = expect(pat, seed, w0 + i);
}
__global__ void check(const uint32_t *d, uint64_t w0, uint64_t nw, int pat, uint32_t seed,
                      unsigned long long *n, unsigned *logged, rec *r) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < nw; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t want = expect(pat, seed, w0 + i), got = d[i];
        if (got != want) {
            atomicAdd(n, 1ull);
            unsigned k = atomicAdd(logged, 1u);
            if (k < CAP) { r[k].w = w0 + i; r[k].got = got; r[k].want = want; }
        }
    }
}

static volatile sig_atomic_t stop_flag = 0;
static void on_sig(int) { stop_flag = 1; }

int main(int argc, char **argv) {
    const double minutes = argc > 1 ? atof(argv[1]) : 10;
    const size_t reserve = (argc > 2 ? atoll(argv[2]) : 1024) << 20;
    const char *seq = argc > 3 ? argv[3] : "40"; const int nseq = (int)strlen(seq);
    const char *pfx = argc > 4 ? argv[4] : "hbm-map";
    const long inject = getenv("HBM_MAP_INJECT") ? atol(getenv("HBM_MAP_INJECT")) : -1;
    signal(SIGTERM, on_sig); signal(SIGINT, on_sig);
    setvbuf(stdout, nullptr, _IOLBF, 0);

    CK(cudaSetDevice(0)); CK(cudaFree(0));
    char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    CUdevice dev; CU(cuCtxGetDevice(&dev));
    char uuid_s[64] = "?"; { CUuuid u; if (cuDeviceGetUuid(&u, dev) == CUDA_SUCCESS) { char *p = uuid_s;
        for (int i = 0; i < 16; i++) p += sprintf(p, "%02x", (unsigned char)u.bytes[i]); } }

    // error buffers first, so they are not carved out of the arena later
    unsigned long long *d_n; unsigned *d_logged; rec *d_rec;
    CK(cudaMalloc(&d_n, 8)); CK(cudaMalloc(&d_logged, 4)); CK(cudaMalloc(&d_rec, CAP * sizeof(rec)));
    std::vector<rec> h_rec(CAP);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = dev;
    size_t G; CU(cuMemGetAllocationGranularity(&G, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t fr, tot; CU(cuMemGetInfo(&fr, &tot));
    const size_t vasz = (tot / G + 1) * G;
    CUdeviceptr base; CU(cuMemAddressReserve(&base, vasz, G, 0, 0));
    std::vector<CUmemGenericAllocationHandle> hs;
    auto ta = std::chrono::steady_clock::now();
    while ((hs.size() + 1) * G <= vasz) {
        CU(cuMemGetInfo(&fr, &tot)); if (fr < reserve + G) break;
        CUmemGenericAllocationHandle h;
        if (cuMemCreate(&h, G, &prop, 0) != CUDA_SUCCESS) break;
        if (cuMemMap(base + hs.size() * G, G, 0, h, 0) != CUDA_SUCCESS) { cuMemRelease(h); break; }
        hs.push_back(h);
    }
    const size_t nh = hs.size(), arena = nh * G;
    CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU(cuMemSetAccess(base, arena, &acc, 1));
    CU(cuMemGetInfo(&fr, &tot));
    printf("device %s uuid %s: granularity %zu KiB, %zu handles = %.2f GiB arena (alloc %.1f s), %.2f GiB left free of %.2f\n",
           bus, uuid_s, G >> 10, nh, arena / 1073741824.0,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count(), fr / 1073741824.0, tot / 1073741824.0);
    printf("pattern sequence %s, %.0f min, logging up to %u bad words per round\n", seq, minutes, CAP);

    char fn[512]; snprintf(fn, sizeof fn, "%s-blocks.csv", pfx);
    FILE *fb = fopen(fn, "w"); if (!fb) { perror(fn); return 2; }
    fprintf(fb, "# device %s uuid %s granularity %zu handles %zu seq %s\n", bus, uuid_s, G, nh, seq);
    fprintf(fb, "round,pattern,t_s,handle,off_in_handle,arena_off,nwords,nzero,or_xor\n");

    const char *names[] = {"hash", "~hash", "check", "~check", "walk"};
    const uint32_t *A = (const uint32_t *)base;
    std::map<size_t, unsigned long long> h_words;           // handle -> bad words
    std::map<size_t, std::set<int>> h_rounds;                // handle -> rounds with errors
    std::map<unsigned long long, int> blk_rounds;            // arena 128B block -> rounds with errors
    unsigned long long total = 0, total_logged = 0; int round = 0, err_rounds = 0, overflow_rounds = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    while (el() < minutes * 60 && !stop_flag) {
        const int pat = seq[round % nseq] - '0'; const uint32_t seed = 0x1234567u * (round + 1);
        for (size_t off = 0; off < arena; off += REGION) {
            size_t len = std::min(REGION, arena - off);
            fill<<<4096, 256>>>((uint32_t *)(A + off / 4), off / 4, len / 4, pat, seed);
        }
        CK(cudaDeviceSynchronize());
        if (inject >= 0 && (size_t)inject < nh)  // self-test only: zero one 128 B block 0x1000 into handle <inject>
            CK(cudaMemset((void *)(base + (size_t)inject * G + 0x1000), 0, 128));
        CK(cudaMemset(d_n, 0, 8)); CK(cudaMemset(d_logged, 0, 4));
        for (size_t off = 0; off < arena; off += REGION) {
            size_t len = std::min(REGION, arena - off);
            check<<<4096, 256>>>(A + off / 4, off / 4, len / 4, pat, seed, d_n, d_logged, d_rec);
        }
        CK(cudaDeviceSynchronize());
        unsigned long long n; unsigned logged;
        CK(cudaMemcpy(&n, d_n, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&logged, d_logged, 4, cudaMemcpyDeviceToHost));
        const double t = el();
        if (n) {
            err_rounds++; total += n;
            unsigned m = std::min(logged, CAP); if (logged > CAP) overflow_rounds++;
            total_logged += m;
            CK(cudaMemcpy(h_rec.data(), d_rec, m * sizeof(rec), cudaMemcpyDeviceToHost));
            struct agg { unsigned nw = 0, nz = 0, ox = 0; };
            std::map<unsigned long long, agg> blocks;
            for (unsigned k = 0; k < m; k++) {
                agg &a = blocks[(h_rec[k].w * 4) >> 7];
                a.nw++; a.nz += h_rec[k].got == 0; a.ox |= h_rec[k].got ^ h_rec[k].want;
            }
            std::set<size_t> hr; int rep_h = 0, rep_b = 0;
            for (auto &kv : blocks) {
                const unsigned long long b = kv.first << 7; const size_t h = b / G;
                fprintf(fb, "%d,%s,%.1f,%zu,0x%zx,0x%llx,%u,%u,%08x\n", round, names[pat], t, h, (size_t)(b % G), b,
                        kv.second.nw, kv.second.nz, kv.second.ox);
                h_words[h] += kv.second.nw;
                if (blk_rounds[kv.first]++ > 0) rep_b++;
                hr.insert(h);
            }
            for (size_t h : hr) { if (!h_rounds[h].empty()) rep_h++; h_rounds[h].insert(round); }
            fflush(fb);
            printf("  round %4d %-6s t=%5.0fs  errors %8llu%s  blocks %5zu (seen before %d)  handles %4zu (seen before %d)  total handles %zu\n",
                   round, names[pat], t, n, logged > CAP ? "+OVF" : "    ", blocks.size(), rep_b, hr.size(), rep_h, h_rounds.size());
        } else if (round % 50 == 0) printf("  round %4d %-6s t=%5.0fs  errors 0\n", round, names[pat], t);
        round++;
    }
    fclose(fb);

    snprintf(fn, sizeof fn, "%s-handles.csv", pfx);
    FILE *fh = fopen(fn, "w");
    std::vector<std::pair<unsigned long long, size_t>> byw;
    if (fh) fprintf(fh, "handle,arena_off,bad_words,rounds\n");
    for (auto &kv : h_words) {
        byw.push_back({kv.second, kv.first});
        if (fh) fprintf(fh, "%zu,0x%zx,%llu,%zu\n", kv.first, kv.first * G, kv.second, h_rounds[kv.first].size());
    }
    if (fh) fclose(fh);
    std::sort(byw.rbegin(), byw.rend());

    printf("\n=== SUMMARY %s ===\n", stop_flag ? "(stopped early)" : "");
    printf("rounds %d, rounds with errors %d (%d overflowed the log), bad words %llu (logged %llu)\n",
           round, err_rounds, overflow_rounds, total, total_logged);
    printf("arena %zu handles of %zu KiB; handles with errors %zu (%.2f%%)\n", nh, G >> 10, h_words.size(), 100.0 * h_words.size() / std::max<size_t>(nh, 1));
    int r1 = 0, r2 = 0, r5 = 0, r10 = 0;
    for (auto &kv : h_rounds) { size_t c = kv.second.size(); r1 += c == 1; r2 += c >= 2; r5 += c >= 5; r10 += c >= 10; }
    printf("handles failing in 1 round %d, >=2 rounds %d, >=5 rounds %d, >=10 rounds %d\n", r1, r2, r5, r10);
    int b2 = 0; for (auto &kv : blk_rounds) b2 += kv.second >= 2;
    printf("distinct bad 128B blocks %zu, of which failed in >=2 rounds %d\n", blk_rounds.size(), b2);
    if (total_logged) {
        for (double frac : {0.5, 0.9, 0.99, 1.0}) {
            unsigned long long acc_w = 0; size_t k = 0;
            while (k < byw.size() && acc_w < frac * total_logged) acc_w += byw[k++].first;
            printf("  covering %3.0f%% of logged bad words needs %zu handles = %.1f MiB quarantined\n", frac * 100, k, k * (G / 1048576.0));
        }
        printf("top handles (handle: bad words / rounds):");
        for (size_t k = 0; k < byw.size() && k < 15; k++) printf(" %zu:%llu/%zu", byw[k].second, byw[k].first, h_rounds[byw[k].second].size());
        printf("\n");
        // where inside a handle do errors land? 16 buckets of G/16
        unsigned long long hist[16] = {};
        FILE *fr2 = fopen((std::string(pfx) + "-blocks.csv").c_str(), "r"); char line[256];
        if (fr2) { while (fgets(line, sizeof line, fr2)) { if (line[0] == '#' || line[0] == 'r') continue;
            int rr; char pn[16]; double tt; size_t hh, oo; if (sscanf(line, "%d,%15[^,],%lf,%zu,%zx", &rr, pn, &tt, &hh, &oo) == 5) hist[oo * 16 / G]++; } fclose(fr2); }
        printf("bad blocks by position inside handle (16ths):"); for (int i = 0; i < 16; i++) printf(" %llu", hist[i]); printf("\n");
    }
    printf("RESULT: %s\n", total ? (r2 == 0 ? "ERRORS, NO handle repeated (wandering)" : "ERRORS, see repeat counts") : "PASS");
    return 0;
}
