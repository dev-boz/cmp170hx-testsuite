#!/bin/bash
# Verdict tests for cmp-quick, with no GPU: a fake nvidia-smi, journalctl and test programs, one scenario each.
#   quick/test.sh [scenario-name]
# Checks the exit status and that the expected words reach the screen and the report.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
T=$(mktemp -d); trap '[[ -n ${KEEP:-} ]] && echo "kept $T" || rm -rf "$T"' EXIT
K=$T/kit; mkdir -p "$K/bin" "$T/path" "$T/sys/0000:43:00.0" "$T/sys/0000:42:00.0" "$T/out" "$T/home"
cp "$HERE/cmp-quick" "$HERE/run-tool" "$HERE/run-memtest" "$K/"; touch "$K/sm-verify-ga100.bin"
echo 0x10de > "$T/sys/0000:43:00.0/vendor"; echo 0x20c2 > "$T/sys/0000:43:00.0/device"
echo 0x10de > "$T/sys/0000:42:00.0/vendor"; echo 0x20c2 > "$T/sys/0000:42:00.0/device"

cat > "$T/path/nvidia-smi" <<'EOF'
#!/bin/bash
args="$*"
crash=$T_DIR/crashed; [[ -f $crash ]] && { [[ $args == *temperature.gpu,temperature.memory,power* ]] && echo "[GPU requires reset], [N/A], [N/A], [N/A]"; exit 15; }
case $args in
  *--query-compute-apps*) [[ -n ${F_APPS:-} ]] && echo "$F_UUID, 1234, miner, 5000 MiB"; exit 0 ;;
  *pci.bus_id,serial,uuid,name*) [[ ${F_NOCARD:-} ]] && exit 0
        echo "00000000:43:00.0, 1000000000001, $F_UUID, NVIDIA CMP 170HX"
        [[ ${F_TWO:-} ]] && echo "00000000:42:00.0, 1000000000002, GPU-owned, NVIDIA CMP 170HX"; exit 0 ;;
  *vbios_version*) echo "92.00.6D.00.0A, ${F_MEM:-65536}, ${F_USED:-14}, 610.57.04, 2, ${F_LW:-16}, 2, 16, 180.00, 40, 50"; exit 0 ;;
  *temperature.gpu,temperature.memory,power*) echo "${F_CORE:-50}, ${F_HBM:-60}, 170.50, 100, ${F_PLNOW:-180.00}"; exit 0 ;;
esac
exit 0
EOF
cat > "$T/path/journalctl" <<'EOF'
#!/bin/bash
[[ ${F_NOLOG:-} ]] && { echo "No journal files were opened due to insufficient permissions." >&2; exit 1; }
for i in $(seq 25); do echo "kernel line $i"; done
[[ ${F_PREXID:-} ]] && echo "NVRM: Xid (PCI:0000:43:00): 79, GPU has fallen off the bus."
[[ -f $T_DIR/xid ]] && echo "NVRM: Xid (PCI:0000:43:00): 119, pid=1, GSP timeout"
exit 0
EOF
printf '#!/bin/bash\n[[ ${F_NOLOG:-} ]] && { echo "dmesg: read kernel buffer failed: Operation not permitted" >&2; exit 1; }\nexit 0\n' > "$T/path/dmesg"
cat > "$K/bin/vram-memtest" <<'EOF'
#!/bin/bash
[[ ${F_DRIVER:-} ]] && { echo "CUDA error CUDA driver version is insufficient for CUDA runtime version at line 60"; exit 3; }
[[ ${F_NOSTART:-} ]] && { echo "./vram-memtest: error while loading shared libraries: libcuda.so.1: cannot open shared object file"; exit 127; }
echo "device 0000:43:00.0: testing ${F_GIB:-63} GiB of ${F_TOT:-64.0} GiB for 3 min (hold 0 s every 4th round), pattern sequence 40"
if [[ ${F_HBMBAD:-} ]]; then
  echo "  round   5 hash        t=    1s  errors 2640  <<<"; echo "      chunk  1 word 123 (byte 0x1ec)  got 00000000 want 12345678 xor 12345678"; sleep 30; exit 1
fi
[[ ${F_MIDCRASH:-} ]] && { sleep 1; echo "CUDA error unknown error at line 77"; exit 3; }
[[ ${F_CRASH:-} ]] && { sleep 1; touch "$T_DIR/crashed" "$T_DIR/xid"; sleep 30; exit 3; }
[[ ${F_SLOW:-} ]] && sleep 3
echo "rounds 12, 1500 GB written+verified, total errors 0"; echo "RESULT: PASS"
EOF
cat > "$K/bin/tc-stress" <<'EOF'
#!/bin/bash
echo "NVIDIA CMP 170HX at 0000:43:00.0: 74 SMs, mode both, 45 s"
if [[ ${F_TCBAD:-} ]]; then echo "  t=  10s  checks 100  errors 7"; sleep 30; exit 1; fi
echo "SMs exercised: 74, SMs with errors: 0, total errors: 0"; echo "RESULT: PASS"
EOF
cat > "$K/bin/sm-verify" <<'EOF'
#!/bin/bash
echo "blocks ran on 74 distinct SM ids; 0 run-to-run mismatches over 5 reps"
if [[ ${F_SMBAD:-} ]]; then echo "vs baseline: 12 / 16384 blocks differ"; echo "RESULT: FAIL - miscomputation detected"; exit 1; fi
echo "vs baseline: 0 / 16384 blocks differ"; echo "RESULT: PASS - bit-identical on every SM"
EOF
cat > "$K/bin/gpu_burn" <<'EOF'
#!/bin/bash
[[ ${F_BURNGONE:-} ]] && exit 127
[[ ${F_BURNJIT:-} ]] && { echo "Couldn't init a GPU test: Error in load module (gpu_burn-drv.cpp:244): PTX JIT compiler library not found"; echo "No clients are alive!  Aborting"; exit 123; }
[[ ${F_BURNDEAD:-} ]] && { echo "0.0%  proc'd: -1 (0 Gflop/s)   errors: 0  (DIED!)  temps: 47 C"; echo "No clients are alive!  Aborting"; exit 123; }
if [[ ${F_BURNBAD:-} ]]; then printf '10.0%%  proc: 1 (20 Gflop/s)   errors: 660000000  (WARNING!)   temps: 40 C\r'; sleep 30; exit 1; fi
if [[ ${F_BURNDIED:-} ]]; then printf '10.0%%  proc: 1 (20 Gflop/s)   errors: 0  (DIED!)   temps: 40 C\n'; echo "GPU 0: FAULTY"; exit 1; fi
printf '100.0%%  proc: 1 (20 Gflop/s)   errors: 0   temps: 60 C\r\nTested 1 GPUs:\n\tGPU 0: OK\n'
EOF
mkdir -p "$T/cublas"; touch "$T/cublas/libcublas.so.12"
cat > "$K/bin/memtest_vulkan" <<'EOF'
#!/bin/bash
# like the real one: lists devices, reads the number only on a terminal, otherwise tests device 1
echo "1: Bus=0x${F_MVFIRST:-11}:00 DevId=0x20C2   64GB NVIDIA CMP 170HX"
[[ ${F_MVUNLISTED:-} ]] || echo "2: Bus=0x43:00 DevId=0x20C2   64GB NVIDIA CMP 170HX"
if [[ -t 0 && -z ${F_MVWRONG:-} ]]; then printf '(first device will be autoselected in 9 seconds)   Override index to test:'; read -r n; else n=1; fi
[[ $n == 2 ]] && bus=43 || bus=${F_MVFIRST:-11}
echo "Standard 5-minute test of $n: Bus=0x$bus:00 DevId=0x20C2   64GB NVIDIA CMP 170HX"
echo "      1 iteration. Passed  0.15 seconds"
if [[ ${F_MVBAD:-} ]]; then echo "Error found. Mode NEXT_RE_READ, total errors 0x20 out of 0x2C000000"; sleep 60; exit 1; fi
sleep 1; echo "Standard 5-minute test PASSed! Just press Ctrl+C unless you plan long test run."; sleep 60
EOF
cat > "$T/fake-wb" <<'EOF'
#!/bin/bash
[[ -r $1 ]] || { echo "open O_DIRECT: No such file or directory"; exit 1; }
if [[ ${F_AIMISMATCH:-} ]]; then echo "  MISMATCH blk.0.attn  CUDA0 tensor+128 file@4096 page_off 0  file 01 vram 00 xor 01"; sleep 60; exit 1; fi
tok=" Paris"; [[ ${F_AIWRONG:-} ]] && tok=" London"
nan=0; [[ ${F_AINAN:-} ]] && nan=151936
echo "logits: $nan non-finite of 151936; top token '$tok'"
echo "weights compared: 218 tensors, 0.39 GB (CPU-resident or size-mismatched skipped: 1)"
echo "corrupted: 0 tensors, 0 bytes"; echo "RESULT: all GPU weights match the file"
EOF
mkdir -p "$T/x/extra/lib" "$T/x/extra/bin" "$T/extras"
touch "$T/x/extra/lib/libcublas.so.12"; cp "$T/fake-wb" "$T/x/extra/bin/weight-readback"; chmod +x "$T/x/extra/bin/weight-readback"
tar czf "$T/extras/cmp-quick-extras.tgz" -C "$T/x" extra
echo model > "$T/extras/small.gguf"; (cd "$T/extras" && sha256sum small.gguf > SHA256SUMS)
mkdir -p "$T/extras-bad"; cp "$T/extras"/* "$T/extras-bad/"; echo damaged > "$T/extras-bad/small.gguf"
mkdir -p "$T/extras-bad2"; cp "$T/extras"/* "$T/extras-bad2/"; echo notgzip > "$T/extras-bad2/cmp-quick-extras.tgz"
chmod +x "$T"/path/* "$K"/bin/* "$K/cmp-quick" "$K/run-tool" "$K/run-memtest"

pass=0 fail=0 only=${1:-}
# scenario <name> <want-exit> <want-text> [ENV=... ...] [-- args]
scenario() {
    local name=$1 want=$2 text=$3; shift 3
    [[ -n $only && $only != "$name" ]] && return
    local envs=() args=(); while (( $# )); do [[ $1 == -- ]] && { shift; args=("$@"); break; }; envs+=("$1"); shift; done
    rm -f "$T/crashed" "$T/xid"; rm -rf "$T/out"/*
    local out rc
    pkill -KILL -f "$T/kit" 2>/dev/null
    out=$(env -i PATH="$T/path:/usr/bin:/bin" HOME="$T/home" T_DIR="$T" F_UUID=GPU-test CMP_QUICK_NOLOADER=1 CMP_SYSFS="$T/sys" CMP_QUICK_OUT="$T/out" CMP_QUICK_CUBLAS_DIRS="$T/none" \
          "${envs[@]}" timeout 90 "$K/cmp-quick" "${args[@]}" 2>&1 </dev/null); rc=$?
    [[ -n ${SHOW:-} ]] && printf '%s\n' "$out"
    if [[ $rc == "$want" ]] && grep -qF "$text" <<< "$out"; then pass=$((pass + 1)); echo "ok    $name (exit $rc)"
    else fail=$((fail + 1)); echo "FAIL  $name: exit $rc want $want, looking for: $text"; tail -25 <<< "$out" | sed 's/^/      /'; fi
}

scenario good           0 "PASS ██"
scenario all-extras     0 "answered Paris"                                  CMP_QUICK_EXTRAS="$T/extras"
scenario extras-burn    0 "60 s, 0 errors"                                  CMP_QUICK_EXTRAS="$T/extras"
scenario ai-mismatch    1 "weights in VRAM differ"                          CMP_QUICK_EXTRAS="$T/extras" F_AIMISMATCH=1
scenario ai-nan         1 "151936 non-finite"                               CMP_QUICK_EXTRAS="$T/extras" F_AINAN=1
scenario ai-wrong       1 "Do not pay"                                      CMP_QUICK_EXTRAS="$T/extras" F_AIWRONG=1
scenario model-damaged  0 "failed its checksum"                             CMP_QUICK_EXTRAS="$T/extras-bad"
scenario pack-damaged   0 "extras pack is damaged"                          CMP_QUICK_EXTRAS="$T/extras-bad2"
scenario no-extras      0 "extras folder isn't next to the kit"
scenario mv-off-default 0 "gpu-burn"
scenario mv-pass        0 "standard 5-minute test on"                         -- --vulkan
scenario mv-right-card  0 "Bus=0x43"                                        -- --vulkan
scenario mv-errors      1 "errors found"                                    F_MVBAD=1 -- --vulkan
scenario mv-wrong-card  0 "chose the wrong card"                            F_MVWRONG=1 -- --vulkan
scenario mv-not-listed  0 "does not list a card on bus 43"                  F_MVUNLISTED=1 -- --vulkan
scenario id-only        0 "NOT TESTED YET"                                  -- --id
scenario hbm-errors     1 "memory errors"                                   F_HBMBAD=1
scenario hbm-stops-fast 1 "Do not pay"                                      F_HBMBAD=1
scenario tensor-errors  1 "errors 7"                                        F_TCBAD=1
scenario sm-mismatch    1 "12 / 16384 blocks differ"                        F_SMBAD=1
scenario crash          1 "stopped answering"                               F_CRASH=1
scenario crash-xid      1 "Xid 119"                                         F_CRASH=1
scenario midrun-cuda    1 "CUDA error unknown error"                        F_MIDCRASH=1
scenario new-xid-only   2 "earlier this boot"                               F_PREXID=1
scenario capped-midrun  2 "power limit cut from 180 W to 100 W"             F_PLNOW=100.00 F_SLOW=1
scenario cap-not-fail   2 "INCONCLUSIVE"                                    F_PLNOW=100.00 F_SLOW=1
scenario hot-core       2 "too hot"                                         F_CORE=72 F_SLOW=1
scenario hot-hbm        2 "HBM 85C"                                         F_HBM=85 F_SLOW=1
scenario stock-8gb      2 "not running the unlock"                          F_MEM=8192 F_GIB=6 F_TOT=8.0
scenario stock-8gb-hbm  2 "the rest of the 64 GB is untested"               F_MEM=8192 F_GIB=6 F_TOT=8.0
scenario stock-id       2 "unlock not running here"                         F_MEM=8192 -- --id
scenario short-alloc    2 "only 40 GiB of 64"                               F_GIB=40
scenario old-driver     2 "could not start"                                 F_DRIVER=1
scenario no-libcuda     2 "could not start"                                 F_NOSTART=1
scenario card-busy      2 "card is in use"                                  F_APPS=1
scenario vram-used      2 "card is in use"                                  F_USED=9000
scenario no-kernel-log  0 "kernel log unreadable"                           F_NOLOG=1
scenario no-card-driver 2 "driver is not running it"                        F_NOCARD=1
scenario x4-note        0 "PCIe x4"                                         F_LW=4
scenario two-no-tty     64 "run again with --serial"                        F_TWO=1
scenario two-by-serial  0 "PASS ██"                                         F_TWO=1 -- --serial 1000000000001
scenario wrong-serial   64 "no CMP 170HX with serial"                       -- --serial 999
scenario no-cublas-skip 0 "no cuBLAS here"
scenario readonly-stick 0 "home/cmp-quick-results"                          CMP_QUICK_OUT=/proc/nope
scenario burn-ok        0 "60 s, 0 errors"                                  CMP_QUICK_CUBLAS_DIRS="$T/cublas"
scenario burn-errors    1 "660000000"                                       CMP_QUICK_CUBLAS_DIRS="$T/cublas" F_BURNBAD=1
scenario burn-faulty    1 "Do not pay"                                      CMP_QUICK_CUBLAS_DIRS="$T/cublas" F_BURNDIED=1
scenario burn-no-jit    2 "could not start"                                 CMP_QUICK_CUBLAS_DIRS="$T/cublas" F_BURNJIT=1
scenario burn-card-died 1 "No clients are alive"                            CMP_QUICK_CUBLAS_DIRS="$T/cublas" F_BURNDEAD=1
scenario kit-tool-gone  2 "could not start"                                 CMP_QUICK_CUBLAS_DIRS="$T/cublas" F_BURNGONE=1

# owned serials skip your own cards
if [[ -z $only || $only == owned ]]; then
    mkdir -p "$T/home/.config/cmp-quick"; echo 1000000000002 > "$T/home/.config/cmp-quick/owned-serials"
    scenario owned 0 "SN 1000000000001" F_TWO=1
    rm -rf "$T/home/.config"
fi
# the report must exist, without colour codes
if [[ -z $only || $only == report ]]; then
    rm -rf "$T/out"/*
    env -i PATH="$T/path:/usr/bin:/bin" HOME="$T/home" T_DIR="$T" F_UUID=GPU-test CMP_QUICK_NOLOADER=1 CMP_SYSFS="$T/sys" CMP_QUICK_OUT="$T/out" CMP_QUICK_CUBLAS_DIRS="$T/none" \
        "$K/cmp-quick" >/dev/null 2>&1 </dev/null
    r=$(ls "$T"/out/cmp-quick-results/*/report.txt 2>/dev/null | head -1)
    if [[ -s $r ]] && grep -q 'PASS ██' "$r" && ! grep -q $'\e' "$r"; then pass=$((pass + 1)); echo "ok    report"; else fail=$((fail + 1)); echo "FAIL  report: $r"; fi
fi
echo; echo "$pass passed, $fail failed"; (( fail == 0 ))
