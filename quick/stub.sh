#!/bin/bash
# cmp-quick.run: self-extracting CMP 170HX quick check. Run it with:  bash cmp-quick.run  [--id|--long|--serial N|--help]
# Unpacks to a temporary folder, runs, then deletes that folder. Nothing is installed.
# Results go in cmp-quick-results/ next to this file (or in $HOME or /tmp if this folder is read-only).
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"   # started with sh
set -u
self=$(readlink -f "$0")
payload_line=$(awk '/^__PAYLOAD_BELOW__$/ { print NR + 1; exit }' "$self")
want=@@SHA256@@
if command -v sha256sum >/dev/null; then
    have=$(tail -n +"$payload_line" "$self" | sha256sum | cut -d' ' -f1)
    if [[ $have != "$want" ]]; then
        echo "cmp-quick.run is damaged (checksum mismatch): copy it again from a good source." >&2; exit 3
    fi
fi
# The tools must execute from where they unpack, and some machines mount /tmp noexec.
dir=""
for base in "${TMPDIR:-/tmp}" /tmp /dev/shm "${XDG_RUNTIME_DIR:-}" "$HOME"; do
    [[ -n $base && -d $base && -w $base ]] || continue
    d=$(mktemp -d "$base/cmp-quick.XXXXXX" 2>/dev/null) || continue
    printf '#!/bin/sh\nexit 0\n' > "$d/x"; chmod +x "$d/x" 2>/dev/null
    if "$d/x" 2>/dev/null; then rm -f "$d/x"; dir=$d; break; fi
    rm -rf "$d"
done
[[ -n $dir ]] || { echo "No writable folder here allows running programs (tried /tmp, /dev/shm, \$HOME)." >&2; exit 3; }
trap 'rm -rf "$dir"' EXIT
tail -n +"$payload_line" "$self" | tar xzf - --no-same-owner -C "$dir" || { echo "could not unpack" >&2; exit 3; }
out=$(dirname "$self")
CMP_QUICK_OUT=$out CMP_QUICK_EXTRAS=$out/extras "$dir/cmp-quick/cmp-quick" "$@"
exit $?
__PAYLOAD_BELOW__
