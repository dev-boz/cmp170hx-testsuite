// weight-readback: load a GGUF with llama.cpp exactly as llama-server does (mmap, layer split over
// every visible GPU), run one short prompt, and in the eval callback read every GPU-resident weight
// tensor back from VRAM and compare it byte-for-byte with the file. The reference is read with
// O_DIRECT so it bypasses the page cache (a bad host-RAM cell cannot fake a match or a mismatch).
//
//   weight-readback <model.gguf> [n_gpu_layers=99]
// Pick cards with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 (split) or =0 (control).
//
// Per mismatching byte it logs: tensor, device buffer, file offset, file offset mod 4096,
// file byte, VRAM byte, XOR. A host-RAM fault shows as single bytes at one fixed page offset;
// an HBM fault shows as runs of wrong bytes (whole 128-byte bursts).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

struct state {
    int fd = -1;
    gguf_context * gg = nullptr;
    size_t data_off = 0;
    std::set<std::string> seen;
    size_t tensors = 0, bytes = 0, bad_tensors = 0, bad_bytes = 0, skipped = 0;
    std::map<std::string, size_t> bad_by_buf;       // device buffer name -> bad bytes
    std::map<int, size_t> bad_by_page_off;          // file offset % 4096 -> count
    std::map<int, size_t> bad_by_xor;               // xor value -> count
    int printed = 0;
};

// O_DIRECT read of [off, off+len) into dst
static bool direct_read(int fd, size_t off, size_t len, uint8_t * dst) {
    const size_t A = 4096, start = off & ~(A - 1), end = (off + len + A - 1) & ~(A - 1);
    const size_t CH = 64u << 20;
    void * buf = nullptr;
    if (posix_memalign(&buf, A, CH)) return false;
    size_t pos = start;
    while (pos < end) {
        size_t n = std::min(CH, end - pos);
        ssize_t r = pread(fd, buf, n, (off_t)pos);
        if (r <= 0) { free(buf); return false; }
        // copy the overlap of [pos, pos+r) with [off, off+len)
        size_t lo = std::max(pos, off), hi = std::min(pos + (size_t)r, off + len);
        if (hi > lo) memcpy(dst + (lo - off), (uint8_t *)buf + (lo - pos), hi - lo);
        pos += (size_t)r;
        if ((size_t)r < n) break;   // EOF
    }
    free(buf);
    return true;
}

static void check_tensor(state & st, const ggml_tensor * w) {
    if (!w || !w->buffer || w->view_src) return;
    const std::string name = ggml_get_name(w);
    if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) return;
    if (!st.seen.insert(name).second) return;
    if (ggml_backend_buffer_is_host(w->buffer)) { st.skipped++; return; }   // CPU-resident: not the question
    const int64_t id = gguf_find_tensor(st.gg, name.c_str());
    if (id < 0) { printf("  %s: not in file\n", name.c_str()); return; }
    const size_t n = ggml_nbytes(w), fsz = gguf_get_tensor_size(st.gg, id);
    if (n != fsz) { printf("  %s: size differs (VRAM %zu vs file %zu), skipped\n", name.c_str(), n, fsz); st.skipped++; return; }
    const size_t foff = st.data_off + gguf_get_tensor_offset(st.gg, id);
    std::vector<uint8_t> vram(n), file(n);
    ggml_backend_tensor_get(w, vram.data(), 0, n);
    if (!direct_read(st.fd, foff, n, file.data())) { printf("  %s: file read failed\n", name.c_str()); return; }
    st.tensors++; st.bytes += n;
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (vram[i] == file[i]) continue;
        bad++;
        const size_t fo = foff + i;
        st.bad_by_page_off[(int)(fo % 4096)]++;
        st.bad_by_xor[vram[i] ^ file[i]]++;
        if (st.printed < 40) {
            printf("  MISMATCH %-32s %-6s tensor+%-10zu file@%-12zu page_off %4zu  file %02x vram %02x xor %02x\n",
                   name.c_str(), ggml_backend_buffer_name(w->buffer), i, fo, fo % 4096, file[i], vram[i], vram[i] ^ file[i]);
            st.printed++;
        }
    }
    if (bad) { st.bad_tensors++; st.bad_bytes += bad; st.bad_by_buf[ggml_backend_buffer_name(w->buffer)] += bad; }
}

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    if (ask) return true;
    state & st = *(state *)ud;
    for (int i = 0; i < GGML_MAX_SRC; i++) check_tensor(st, t->src[i]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: weight-readback <model.gguf> [n_gpu_layers]\n"); return 2; }
    const char * path = argv[1];
    state st;
    st.fd = open(path, O_RDONLY | O_DIRECT);
    if (st.fd < 0) { perror("open O_DIRECT"); return 1; }
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    st.gg = gguf_init_from_file(path, gp);
    if (!st.gg) { fprintf(stderr, "gguf parse failed (split GGUFs: pass a single-file model)\n"); return 1; }
    st.data_off = gguf_get_data_offset(st.gg);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 2 ? atoi(argv[2]) : 99;
    mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb; cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "The capital of France is";
    std::vector<llama_token> toks(64);
    int nt = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, false);
    toks.resize(nt);
    printf("decoding %d prompt tokens and reading back weights...\n", nt); fflush(stdout);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), nt))) { fprintf(stderr, "decode failed\n"); return 1; }

    const float * lg = llama_get_logits_ith(ctx, -1);
    const int nv = llama_vocab_n_tokens(vocab);
    int nan = 0, best = 0;
    for (int i = 0; i < nv; i++) { if (!std::isfinite(lg[i])) nan++; else if (lg[i] > lg[best]) best = i; }
    char piece[64] = {0};
    llama_token_to_piece(vocab, best, piece, sizeof piece - 1, 0, false);

    printf("\nlogits: %d non-finite of %d; top token '%s'\n", nan, nv, piece);
    printf("weights compared: %zu tensors, %.2f GB (CPU-resident or size-mismatched skipped: %zu)\n",
           st.tensors, st.bytes / 1e9, st.skipped);
    printf("corrupted: %zu tensors, %zu bytes\n", st.bad_tensors, st.bad_bytes);
    for (auto & kv : st.bad_by_buf) printf("  by buffer %s: %zu bytes\n", kv.first.c_str(), kv.second);
    if (st.bad_bytes) {
        printf("  by file-offset%%4096 (top):"); int k = 0;
        std::vector<std::pair<size_t,int>> v; for (auto & kv : st.bad_by_page_off) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend()); for (auto & p : v) { if (k++ == 10) break; printf(" %d:%zu", p.second, p.first); } printf("\n");
        printf("  by xor (top):"); k = 0; v.clear(); for (auto & kv : st.bad_by_xor) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend()); for (auto & p : v) { if (k++ == 10) break; printf(" %02x:%zu", p.second, p.first); } printf("\n");
    }
    printf("RESULT: %s\n", st.bad_bytes ? "VRAM WEIGHTS DIFFER FROM FILE" : "all GPU weights match the file");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return st.bad_bytes ? 1 : 0;
}
