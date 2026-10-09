#!/bin/bash
# Build the CUDA tools and link the cmp-* commands into ~/.local/bin (or $PREFIX/bin).
#   ./install.sh                         CUDA tools only
#   LLAMA_DIR=~/llama.cpp ./install.sh   also build weight-readback for cmp-ai-test
#   ./install.sh remove                  remove the links
set -e
ROOT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); DEST=${PREFIX:-$HOME/.local}/bin
CMDS="identity hbm-test tensor-test sm-test ai-test hbm-map kernel-check summary"
if [[ ${1:-} == remove ]]; then
    for c in $CMDS telemetry; do rm -f "$DEST/cmp-$c"; done; echo "removed cmp-* links from $DEST"; exit 0
fi
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
make -C "$ROOT"
[[ -n ${LLAMA_DIR:-} ]] && make -C "$ROOT" weight-readback LLAMA_DIR="$LLAMA_DIR"
mkdir -p "$DEST"
for c in $CMDS; do ln -sf "$ROOT/scripts/cmptest" "$DEST/cmp-$c"; done
ln -sf "$ROOT/scripts/telemetry.sh" "$DEST/cmp-telemetry"
chmod +x "$ROOT"/scripts/*.sh "$ROOT/scripts/cmptest" "$ROOT/scripts/model-cache-check.py"
echo "installed: $(for c in $CMDS telemetry; do printf 'cmp-%s ' $c; done)-> $DEST"
[[ :$PATH: == *":$DEST:"* ]] || echo "note: $DEST is not on your PATH"
