#!/bin/bash
# Build the buy-in-person kit: one self-extracting cmp-quick.run plus extras, ready to copy to a USB stick.
#   quick/build.sh [out-dir=~/cmp-quick-kit]
# Needs bin/ built (make weight-readback LLAMA_DIR=...), and three variables: OFFSHELF=<folder with gpu-burn/ and
# memtest_vulkan>, LLAMA_LIB=<llama.cpp>/build/bin, MODEL=<small .gguf>. See quick/README.txt, "Building the kit".
# The output holds third-party binaries (glibc, libstdc++, gpu-burn, memtest_vulkan): it stays out of git.
set -eu
: "${OFFSHELF:?set OFFSHELF=<folder holding gpu-burn/ (built) and memtest_vulkan>}"
: "${LLAMA_LIB:?set LLAMA_LIB=<llama.cpp>/build/bin (built with -DGGML_CUDA=ON)}"
: "${MODEL:?set MODEL=<a small single-file .gguf, e.g. a 0.5B Q4 model>}"
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); REPO=$(dirname "$HERE")
OUT=${1:-$HOME/cmp-quick-kit}
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
K=$stage/cmp-quick; mkdir -p "$K/bin" "$K/lib"

# Check the system libraries we are about to copy are intact (a bad RAM cell can corrupt files on non-ECC machines).
for pkg in libc6 libstdc++6 libgcc-s1 libgomp1 libvulkan1; do
    bad=$(dpkg --verify "$pkg:amd64" 2>/dev/null || dpkg --verify "$pkg" 2>/dev/null || true)
    [[ -z $bad ]] || { echo "dpkg --verify $pkg reports: $bad"; exit 1; }
done

cp "$HERE/cmp-quick" "$HERE/run-tool" "$HERE/run-memtest" "$K/"; chmod +x "$K/cmp-quick" "$K/run-tool" "$K/run-memtest"
cp "$REPO/baselines/sm-verify-ga100.bin" "$K/"
for t in vram-memtest tc-stress sm-verify; do cp "$REPO/bin/$t" "$K/bin/"; done
cp "$OFFSHELF/gpu-burn/gpu_burn" "$OFFSHELF/memtest_vulkan" "$K/bin/"
# gpu-burn's own build ships this kernel as PTX only, which needs the driver's JIT library. Precompile it for the GA100.
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
"$NVCC" -fatbin -O3 -gencode arch=compute_80,code=sm_80 -o "$K/bin/compare.fatbin" "$OFFSHELF/gpu-burn/compare.cu"
for l in ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 libgcc_s.so.1 libstdc++.so.6 libvulkan.so.1; do
    cp -L "/lib/x86_64-linux-gnu/$l" "$K/lib/"
done

tar czf "$stage/payload.tgz" -C "$stage" cmp-quick
sum=$(sha256sum "$stage/payload.tgz" | cut -d' ' -f1)
mkdir -p "$OUT/extras"
sed "s/@@SHA256@@/$sum/" "$HERE/stub.sh" > "$stage/run"
cat "$stage/payload.tgz" >> "$stage/run"
cp "$stage/run" "$OUT/.cmp-quick.run.new"; chmod +x "$OUT/.cmp-quick.run.new"; mv "$OUT/.cmp-quick.run.new" "$OUT/cmp-quick.run"
cp "$HERE/README.txt" "$HERE/cmp-telemetry.sh" "$OUT/"
cp "$OFFSHELF/memtest_vulkan" "$OUT/extras/memtest_vulkan"
[[ -f $OFFSHELF/memtest_vulkan-v0.5.0.exe ]] && cp "$OFFSHELF/memtest_vulkan-v0.5.0.exe" "$OUT/extras/"

# The extras pack: cuBLAS for gpu-burn and llama.cpp, plus the AI test and a model small enough to read off a stick.
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/targets/x86_64-linux/lib}
X=$stage/x/extra; mkdir -p "$X/lib" "$X/bin"
cp -L "$CUDA_LIB"/libcublas.so.12 "$CUDA_LIB"/libcublasLt.so.12 "$CUDA_LIB"/libcudart.so.12 "$X/lib/"
cp -L "$LLAMA_LIB"/libllama.so.0 "$LLAMA_LIB"/libggml.so.0 "$LLAMA_LIB"/libggml-base.so.0 "$LLAMA_LIB"/libggml-cpu.so.0 "$LLAMA_LIB"/libggml-cuda.so.0 "$X/lib/"
cp -L /lib/x86_64-linux-gnu/libgomp.so.1 "$X/lib/"
cp "$REPO/bin/weight-readback" "$X/bin/"
tar czf "$OUT/extras/.pack.new" -C "$stage/x" extra; mv "$OUT/extras/.pack.new" "$OUT/extras/cmp-quick-extras.tgz"
cp "$MODEL" "$OUT/extras/"; sync "$OUT/extras/$(basename "$MODEL")"
m=$(dd if="$OUT/extras/$(basename "$MODEL")" iflag=direct bs=1M 2>/dev/null | sha256sum | cut -d' ' -f1)
o=$(dd if="$MODEL" iflag=direct bs=1M 2>/dev/null | sha256sum | cut -d' ' -f1)
[[ $m == "$o" ]] || { echo "model copy differs from the original, RAM or disk fault: build again"; exit 1; }
echo "$m  $(basename "$MODEL")" > "$OUT/extras/SHA256SUMS"
echo "$OUT/extras  $(du -sh "$OUT/extras" | cut -f1)"

# Read the result straight off the disk, past the page cache, and check it against what was built.
a=$(sha256sum "$stage/run" | cut -d' ' -f1)
sync "$OUT/cmp-quick.run"
b=$(dd if="$OUT/cmp-quick.run" iflag=direct bs=1M 2>/dev/null | sha256sum | cut -d' ' -f1)
[[ $a == "$b" ]] || { echo "written file differs from the build, RAM or disk fault: build again"; exit 1; }
echo "$OUT/cmp-quick.run  $(du -h "$OUT/cmp-quick.run" | cut -f1)  sha256 $a"
