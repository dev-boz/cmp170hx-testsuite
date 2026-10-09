#!/bin/bash
# Live telemetry for the card under test, redrawn every second (run it in a second terminal pane).
# Read-only: nvidia-smi, the blower tach if configured (config.example.sh), the kernel log.
source "$(dirname "$(readlink -f "$0")")/common.sh"
find_card || { sleep 30; exit 1; }
start=${TELEMETRY_START:-$(date +%s)}; short=${CARD_BDF#0000:}; xids=0; tick=0
tput civis 2>/dev/null; clear
trap 'tput cnorm 2>/dev/null; exit' INT TERM
tc() { [[ $1 =~ ^[0-9]+$ ]] || { printf "${RED}%s${N}" "$1"; return; }
       if (( $1 >= $3 )); then printf "${RED}%s°C${N}" "$1"; elif (( $1 >= $2 )); then printf "${YEL}%s°C${N}" "$1"; else printf "${GRN}%s°C${N}" "$1"; fi; }
while :; do
    q=$(timeout 5 nvidia-smi -i "$CARD_IDX" --format=csv,noheader,nounits --query-gpu=temperature.gpu,temperature.memory,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current 2>&1)
    IFS=',' read -r gt mt pw pl ut mu mtot sm mm lg lw <<< "${q// /}"
    fan=""; [[ -n $BLOWER_TACH ]] && fan="   Blower ($BLOWER_NAME) ${B}$(cat "$BLOWER_TACH" 2>/dev/null)${N} rpm"
    (( tick++ % 5 == 0 )) && xids=$(xid_count)
    xc=$([[ $xids == 0 ]] && echo "${GRN}0${N}" || echo "${RED}$xids${N}")
    now=$(date +%s)
    tput cup 0 0
    printf " ${CYN}${B}LIVE TELEMETRY${N}  %s   elapsed ${B}%s${N}\e[K\n" "$(date '+%a %d %b %Y  %H:%M:%S')" "$(hms $((now - start)))"
    printf " ${B}%s${N}  SN ${B}%s${N}  PCI %s (%s)  PCIe Gen%s x%s\e[K\n" "$CARD_NAME" "$CARD_SERIAL" "$CARD_BDF" "$CARD_SLOT" "$lg" "$lw"
    printf " Core %b   HBM %b   Power ${B}%.0f${N} / %.0f W   GPU load ${B}%s%%${N}   VRAM used ${B}%s${N} / %s MiB\e[K\n" \
        "$(tc "$gt" 60 70)" "$(tc "$mt" 70 80)" "${pw:-0}" "${pl:-0}" "$ut" "$mu" "$mtot"
    printf " Clocks: SM %s MHz, HBM %s MHz%s   Kernel GPU errors (Xid): %b\e[K\n" "$sm" "$mm" "$fan" "$xc"
    tput ed 2>/dev/null
    sleep 1
done
