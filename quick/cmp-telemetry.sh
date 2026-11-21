#!/bin/bash
# Live panel for a second terminal while cmp-quick runs: every CMP 170HX, redrawn each second. Ctrl+C to quit.
#   bash cmp-telemetry.sh [--serial N]
# Read-only, needs only bash and nvidia-smi, so it runs from the stick on any Linux machine: no install, no sudo.
# Sources: nvidia-smi, the motherboard fan sensors, the kernel log, and this box's thermal watcher if there is one.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"   # started with sh
set -u
unset LD_PRELOAD LD_LIBRARY_PATH
want=${SERIAL:-}
while (( $# )); do
    case $1 in
        --serial) want=${2:-}; shift ;;
        --serial=*) want=${1#*=} ;;
        -h|--help) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1 (try --help)"; exit 64 ;;
    esac
    shift
done
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: the NVIDIA driver is not installed on this machine."; exit 2; }
HW=${CMP_HWMON:-/sys/class/hwmon}
B=$'\e[1m' DIM=$'\e[2m' RED=$'\e[1;31m' GRN=$'\e[1;32m' YEL=$'\e[1;33m' CYN=$'\e[1;36m' N=$'\e[0m'
declare -A PEAKC=() PEAKM=() SPUN=()
start=$(date +%s) tick=0 KLOG="" KLOG_OK=1

temp() {   # temp <value> <warn> <hot>
    [[ $1 =~ ^[0-9]+$ ]] || { printf '%s%s%s' "$RED" "$1" "$N"; return; }
    if (( $1 >= $3 )); then printf '%s%sC%s' "$RED" "$1" "$N"
    elif (( $1 >= $2 )); then printf '%s%sC%s' "$YEL" "$1" "$N"
    else printf '%s%sC%s' "$GRN" "$1" "$N"; fi
}
read_klog() {   # a user who can't read the log must see "?", never a clean-looking 0
    KLOG=$(journalctl -k -b --no-pager 2>/dev/null)
    (( $(grep -c . <<< "$KLOG") >= 20 )) || KLOG=$(dmesg 2>/dev/null)
    if (( $(grep -c . <<< "$KLOG") >= 20 )); then KLOG_OK=1; else KLOG_OK=0; fi
}
xids() {   # xids <bus id as nvidia-smi prints it>
    (( KLOG_OK )) || { printf '%s?%s (can'"'"'t read the kernel log)' "$YEL" "$N"; return; }
    local bus=$(tr 'A-F' 'a-f' <<< "${1: -12}") n
    n=$(grep -c "Xid (PCI:${bus%.*})" <<< "$KLOG")
    if (( n )); then printf '%s%s%s' "$RED" "$n" "$N"; else printf '%s0%s' "$GRN" "$N"; fi
}
fans() {   # sets FANLINE, not stdout: SPUN must survive, so never call this in a subshell
    # every fan that reads above 0; one that has spun and now reads 0 stays on the panel, in red
    local out="" f name rpm id
    for f in "$HW"/hwmon*/fan*_input; do
        [[ -r $f ]] || continue
        rpm=$(cat "$f" 2>/dev/null); [[ $rpm =~ ^[0-9]+$ ]] || continue
        name=$(cat "$(dirname "$f")/name" 2>/dev/null); name=${name%%_*}
        id="$name $(basename "$f" _input)"
        if (( rpm > 0 )); then SPUN[$id]=1; out+="  $id ${B}$rpm${N}"
        elif [[ -n ${SPUN[$id]:-} ]]; then out+="  ${RED}$id 0 STOPPED${N}"; fi
    done
    FANLINE=${out:-  ${DIM}no fan sensors readable; watch the blower${N}}
}
watcher() {   # this box's thermal watcher, only if it is running
    local st ts lvl; st=$(cat /run/cmp-thermal-watch.state 2>/dev/null) || return
    ts=${st%% *}; [[ $ts =~ ^[0-9]+$ ]] && (( $(date +%s) - ts <= 5 )) || return
    lvl=$(grep -oE "L[0-3] [a-z]+\(${1: -7:2}:" <<< "$st" | cut -c1-2)
    case $lvl in
        L0) printf '   Watcher %snormal%s' "$GRN" "$N" ;;
        L1) printf '   Watcher %sHOT%s' "$YEL" "$N" ;;
        L2|L3) printf '   Watcher %s%s CRITICAL%s' "$RED" "$lvl" "$N" ;;
    esac
}

printf '\e[?25l\e[2J'
trap 'printf "\e[?25h\n"; exit 0' INT TERM
while :; do
    (( tick++ % 5 == 0 )) && read_klog
    printf '\e[H %s%sCMP 170HX LIVE%s  %s   elapsed %s%d:%02d%s\e[K\n' "$CYN" "$B" "$N" "$(date '+%H:%M:%S')" "$B" $(( ($(date +%s) - start) / 60 )) $(( ($(date +%s) - start) % 60 )) "$N"
    rows=$(timeout -k 1 5 nvidia-smi --format=csv,noheader,nounits --query-gpu=name,serial,pci.bus_id,temperature.gpu,temperature.memory,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current 2>/dev/null | grep 'CMP 170HX')
    [[ -n $want ]] && rows=$(grep -F ", $want," <<< "$rows")
    if [[ -z $rows ]]; then
        printf ' %sNo CMP 170HX answering%s%s. If a test was running, the card may have crashed: it needs a full power-off.\e[K\n' "$RED" "$N" "${want:+ with serial $want}"
    fi
    while IFS=',' read -r name ser bus gt mt pw pl ut mu mtot sm mm lg lw; do
        [[ -n ${ser:-} ]] || continue
        ser=${ser// /} bus=${bus// /} gt=${gt// /} mt=${mt// /} pw=${pw// /} pl=${pl// /} ut=${ut// /} mu=${mu// /} mtot=${mtot// /} sm=${sm// /} mm=${mm// /} lg=${lg// /} lw=${lw// /}
        [[ $gt =~ ^[0-9]+$ ]] && (( gt > ${PEAKC[$ser]:-0} )) && PEAKC[$ser]=$gt
        [[ $mt =~ ^[0-9]+$ ]] && (( mt > ${PEAKM[$ser]:-0} )) && PEAKM[$ser]=$mt
        printf '%s──────────────────────────────────────────────────────────────────────%s\e[K\n' "$DIM" "$N"
        printf ' SN %s%s%s  %s  PCIe Gen%s x%s%s\e[K\n' "$B" "$ser" "$N" "${bus: -7}" "$lg" "$lw" "$(watcher "$bus")"
        printf ' Core %s  HBM %s   peak %sC / %sC   Power %s%s%s / %s W   Load %s%s%%%s\e[K\n' \
            "$(temp "$gt" 60 70)" "$(temp "$mt" 70 80)" "${PEAKC[$ser]:-?}" "${PEAKM[$ser]:-?}" "$B" "${pw%.*}" "$N" "${pl%.*}" "$B" "$ut" "$N"
        printf ' VRAM %s / %s MiB   Clocks SM %s, HBM %s MHz   Xid %s\e[K\n' "$mu" "$mtot" "$sm" "$mm" "$(xids "$bus")"
    done <<< "$rows"
    printf '%s──────────────────────────────────────────────────────────────────────%s\e[K\n' "$DIM" "$N"
    fans
    printf ' Fans:%s\e[K\n\e[J' "$FANLINE"
    sleep 1
done
