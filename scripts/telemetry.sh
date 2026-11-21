#!/bin/bash
# Live telemetry, redrawn every second (run it in a second terminal pane). One panel per CMP 170HX;
# SERIAL=... shows just that card.
# Read-only: nvidia-smi, the blower tach if configured, a thermal watcher's state file if one is running, the kernel log.
source "$(dirname "$(readlink -f "$0")")/common.sh"
if [[ -z ${SERIAL:-} ]]; then
    mapfile -t SERIALS < <(nvidia-smi --query-gpu=serial,name --format=csv,noheader 2>/dev/null | awk -F', ' '/CMP 170HX/{print $1}')
else SERIALS=("$SERIAL"); fi
(( ${#SERIALS[@]} )) || { echo "no CMP 170HX found"; sleep 30; exit 1; }
start=${TELEMETRY_START:-$(date +%s)}; tick=0; declare -A XIDS WATCH
tput civis 2>/dev/null; clear
trap 'tput cnorm 2>/dev/null; exit' INT TERM
tc() { [[ $1 =~ ^[0-9]+$ ]] || { printf "${RED}%s${N}" "$1"; return; }
       if (( $1 >= $3 )); then printf "${RED}%s°C${N}" "$1"; elif (( $1 >= $2 )); then printf "${YEL}%s°C${N}" "$1"; else printf "${GRN}%s°C${N}" "$1"; fi; }
# Watcher state -> WATCH[<pci addr>]="level|mode|duty|cap" (empty if it isn't running or the file is stale).
watcher() {
    WATCH=()
    local addr rest
    while IFS=$'\t' read -r addr rest; do WATCH[$addr]=$rest; done < <(python3 -c 'import json, sys, time
try: s = json.load(open(sys.argv[1]))
except (OSError, ValueError): sys.exit()
if time.time() - s["ts"] > 5: sys.exit()
for a, c in s["cards"].items():
    f = s["fans"].get(c["fan"]) or {}
    print(a, "|".join(str(x) for x in (c["level"], s["mode"], f.get("duty", ""), f.get("cap", ""))), sep="\t")' "${CMP_WATCH_STATE:-/run/cmp-thermal-watch.json}" 2>/dev/null)
}
while :; do
    tput cup 0 0
    now=$(date +%s)
    printf " ${CYN}${B}LIVE TELEMETRY${N}  %s   elapsed ${B}%s${N}\e[K\n" "$(date '+%a %d %b %Y  %H:%M:%S')" "$(hms $((now - start)))"
    (( tick++ % 5 == 0 )) && refresh=1 || refresh=0
    watcher
    for SERIAL in "${SERIALS[@]}"; do
        find_card >/dev/null 2>&1 || { printf " ${RED}card SN %s not found${N}\e[K\n" "$SERIAL"; continue; }
        q=$(timeout 5 nvidia-smi -i "$CARD_IDX" --format=csv,noheader,nounits --query-gpu=temperature.gpu,temperature.memory,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current 2>&1)
        IFS=',' read -r gt mt pw pl ut mu mtot sm mm lg lw <<< "${q// /}"
        fan=""; [[ -n $BLOWER_TACH ]] && fan="   Blower ($BLOWER_NAME) ${B}$(cat "$BLOWER_TACH" 2>/dev/null)${N} rpm"
        IFS='|' read -r lvl fmode fduty fcap <<< "${WATCH[$CARD_BDF]:-}"
        case $lvl in
            0) wt="${GRN}normal${N}" ;; 1) wt="${YEL}HOT${N}" ;; 2) wt="${RED}CRITICAL${N}" ;; 3) wt="${RED}EMERGENCY${N}" ;;
            *) wt="${YEL}not running${N}" ;;
        esac
        if [[ -n $fduty ]]; then
            wt+=", fan ${B}$(( fduty * 100 / 255 ))%${N} ($fduty/$fcap max)"
            case $fmode in curve) wt+=" ${GRN}curve${N}" ;; max) wt+=" ${YEL}MAX${N}" ;; all) wt+=" ${RED}ALL FANS MAX${N}" ;; esac
        elif [[ -n $lvl ]]; then wt+=", no fan control"; fi
        (( refresh )) && XIDS[$CARD_IDX]=$(xid_count)
        xids=${XIDS[$CARD_IDX]:-0}
        xc=$([[ $xids == 0 ]] && echo "${GRN}0${N}" || echo "${RED}$xids${N}")
        printf "${DIM}"; line; printf "${N}"
        printf " ${B}%s${N}  SN ${B}%s${N}  PCI %s (%s)  PCIe Gen%s x%s\e[K\n" "$CARD_NAME" "$CARD_SERIAL" "$CARD_BDF" "$CARD_SLOT" "$lg" "$lw"
        printf " Core %b   HBM %b   Power ${B}%.0f${N} / %.0f W   GPU load ${B}%s%%${N}   VRAM used ${B}%s${N} / %s MiB\e[K\n" \
            "$(tc "$gt" 60 70)" "$(tc "$mt" 70 80)" "${pw:-0}" "${pl:-0}" "$ut" "$mu" "$mtot"
        printf " Clocks: SM %s MHz, HBM %s MHz%s   Xid: %b\e[K\n" "$sm" "$mm" "$fan" "$xc"
        printf " Thermal watcher: %b\e[K\n" "$wt"
    done
    # Display / reference GeForce card (if any): found by name, read-only. SHOW_GEFORCE=0 hides it.
    if [[ ${SHOW_GEFORCE:-1} == 1 ]]; then
        q=$(timeout 5 nvidia-smi --format=csv,noheader,nounits --query-gpu=name,pci.bus_id,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,fan.speed,pcie.link.gen.current,pcie.link.width.current 2>/dev/null | grep -m1 'GeForce')
        if [[ -n $q ]]; then
            gn=$(cut -d, -f1 <<< "$q")
            IFS=',' read -r _ gb gt pw pl ut mu mtot sm mm gf lg lw <<< "${q// /}"
            if (( refresh )) || [[ -z ${XGF:-} ]]; then
                x=${gb#0000}; XGF=$(journalctl -k -b --no-pager 2>/dev/null | grep -c "Xid (PCI:${x%.*})")
            fi
            xc=$([[ ${XGF:-0} == 0 ]] && echo "${GRN}0${N}" || echo "${RED}$XGF${N}")
            printf "${DIM}"; line; printf "${N}"
            printf " ${B}%s${N}  PCI %s  PCIe Gen%s x%s\e[K\n" "$gn" "${gb#00000000:}" "$lg" "$lw"
            printf " Core %b   Power ${B}%.0f${N} / %.0f W   GPU load ${B}%s%%${N}   VRAM used ${B}%s${N} / %s MiB\e[K\n" \
                "$(tc "$gt" 70 80)" "${pw:-0}" "${pl:-0}" "$ut" "$mu" "$mtot"
            printf " Clocks: SM %s MHz, Mem %s MHz   Fan ${B}%s%%${N}   Xid: %b\e[K\n" "$sm" "$mm" "$gf" "$xc"
        fi
    fi
    tput ed 2>/dev/null
    sleep 1
done
