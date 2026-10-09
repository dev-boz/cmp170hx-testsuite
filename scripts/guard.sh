#!/bin/bash
# Run a command under a thermal watchdog.
#
#   guard.sh <name> <max-seconds> <command...>
#
# Starts <command> in its own session and kills it (TERM, then KILL after 5 s) if any CMP 170HX core
# reaches GPU_LIMIT, any HBM reaches MEM_LIMIT, nvidia-smi stops answering, a configured blower reads
# below BLOWER_MIN rpm for 3 s (see config.example.sh), or max-seconds elapse. Logs per-second telemetry
# for every CMP to $LOG_DIR/<name>-<time>.csv and prints peak temperatures on exit.
# Exit: the command's status; 99 if guard stopped it for any reason other than the time limit.
set -u
source "$(dirname "$(readlink -f "$0")")/common.sh"
GPU_LIMIT=${GPU_LIMIT:-72}      # card spec: 85 C max operating, slowdown ~95 C
MEM_LIMIT=${MEM_LIMIT:-85}      # HBM spec: 95 C max operating
BLOWER_MIN=${BLOWER_MIN:-800}

name=$1; maxs=$2; shift 2
CMPS=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/CMP 170HX/{print $1}' | paste -sd,)
[[ -n $CMPS ]] || { echo "guard: no CMP 170HX found" >&2; exit 2; }
tachs=(); for bdf in $(cmp_bdfs); do t=$(blower_tach_for "$bdf"); [[ -n $t ]] && tachs+=("$t"); done

mkdir -p "$LOG_DIR"
log=$LOG_DIR/$name-$(date +%Y%m%d-%H%M%S).csv
echo "t,gpu,core_c,hbm_c,power_w,sm_mhz,mem_mhz,util_pct,vram_mib,clk_reasons,blower_rpm" > "$log"

setsid "$@" &
pid=$!
start=$(date +%s); peak=0; peakmem=0; reason=""; lowrpm=0

while kill -0 $pid 2>/dev/null; do
    q=$(timeout 5 nvidia-smi -i "$CMPS" --format=csv,noheader,nounits \
        --query-gpu=index,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,clocks_event_reasons.active)
    if [[ -z $q || $(wc -l <<< "$q") -lt $(tr ',' '\n' <<< "$CMPS" | wc -l) ]]; then reason="telemetry lost"; break; fi
    rpms=""; stall=0
    for t in "${tachs[@]}"; do
        r=$(cat "$t" 2>/dev/null); rpms+="${rpms:+/}$r"
        [[ $r =~ ^[0-9]+$ ]] && (( r < BLOWER_MIN )) && stall=1
    done
    if (( stall )); then lowrpm=$((lowrpm + 1)); else lowrpm=0; fi
    (( lowrpm >= 3 )) && reason="blower stall: $rpms rpm < $BLOWER_MIN"
    el=$(( $(date +%s) - start ))
    while IFS=', ' read -r gi gt mt pw sm mm ut vr cr; do
        echo "$el,$gi,$gt,$mt,$pw,$sm,$mm,$ut,$vr,$cr,$rpms" >> "$log"
        [[ $gt =~ ^[0-9]+$ ]] && (( gt > peak )) && peak=$gt
        [[ $mt =~ ^[0-9]+$ ]] && (( mt > peakmem )) && peakmem=$mt
        if [[ $gt =~ ^[0-9]+$ ]] && (( gt >= GPU_LIMIT )); then reason="GPU$gi ${gt}C >= ${GPU_LIMIT}C"; fi
        if [[ $mt =~ ^[0-9]+$ ]] && (( mt >= MEM_LIMIT )); then reason="GPU$gi HBM ${mt}C >= ${MEM_LIMIT}C"; fi
    done <<< "$q"
    [[ -n $reason ]] && break
    if (( el >= maxs )); then reason="time limit ${maxs}s"; break; fi
    sleep 1
done

if [[ -n $reason ]] && kill -0 $pid 2>/dev/null; then
    echo "guard: STOPPING ($reason)" >&2
    kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null
fi
wait $pid 2>/dev/null; rc=$?
echo "guard: $name finished rc=$rc ${reason:+stopped: $reason; }peak GPU ${peak}C, peak HBM ${peakmem}C, log $log" >&2
[[ -n $reason && $reason != time* ]] && exit 99
exit $rc
