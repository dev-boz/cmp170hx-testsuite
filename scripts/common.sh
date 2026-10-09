#!/bin/bash
# Shared helpers for the CMP 170HX test suite (sourced by cmptest, guard.sh, telemetry.sh).
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
BIN=$ROOT/bin
LOG_DIR=${CMP_LOG_DIR:-$ROOT/logs}     # per-second telemetry CSVs from guard.sh
RUNS_DIR=${CMP_RUNS_DIR:-$ROOT/runs}   # raw output of every test + results.tsv, per card per day
declare -A BLOWERS=()                  # optional, see config.example.sh
[[ -f $ROOT/config.sh ]] && source "$ROOT/config.sh"

B=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[1;31m'; GRN=$'\e[1;32m'; YEL=$'\e[1;33m'; CYN=$'\e[1;36m'; N=$'\e[0m'
W=78   # box width

root_port() { basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$1")")"; }
cmp_bdfs() {   # PCI addresses of every CMP 170HX (10de:20c2 8 GB board, 10de:2082 10 GB board)
    local d
    for d in /sys/bus/pci/devices/*; do
        [[ $(cat "$d/vendor" 2>/dev/null) == 0x10de && $(cat "$d/device" 2>/dev/null) =~ ^0x(20c2|2082)$ ]] && basename "$d"
    done
}
# "it8686:fan2" (hwmon chip-name prefix : tach file) -> /sys/class/hwmon/hwmonN/fan2_input
resolve_tach() {
    local chip=${1%%:*} fan=${1##*:} n
    for n in /sys/class/hwmon/hwmon*/name; do
        [[ $(cat "$n") == "$chip"* ]] && { echo "$(dirname "$n")/${fan}_input"; return; }
    done
}
blower_tach_for() { local spec=${BLOWERS[$(root_port "$1")]:-}; [[ -n $spec ]] && resolve_tach "$spec"; }

# Find the card under test: the single CMP 170HX in the system, or the one whose serial is $SERIAL.
# Sets CARD_IDX (nvidia-smi index = PCI order), CARD_BDF, CARD_SERIAL, CARD_UUID, CARD_NAME,
# CARD_DEVID, CARD_PORT (PCIe root port), CARD_SLOT, BLOWER_TACH/BLOWER_NAME (empty if not configured).
find_card() {
    local rows sel n bus
    rows=$(nvidia-smi --query-gpu=index,pci.bus_id,serial,uuid,name --format=csv,noheader 2>/dev/null | grep 'CMP 170HX')
    if [[ -n ${SERIAL:-} ]]; then sel=$(grep ", $SERIAL," <<< "$rows"); else sel=$rows; fi
    n=$(grep -c . <<< "$sel")
    if (( n != 1 )); then
        echo "${RED}Expected exactly one CMP 170HX${SERIAL:+ with serial $SERIAL}, found $n:${N}"; echo "$rows"
        echo "(test one card at a time; set SERIAL=<serial> to pick one)"; return 1
    fi
    IFS=',' read -r CARD_IDX bus CARD_SERIAL CARD_UUID CARD_NAME <<< "$sel"
    CARD_SERIAL=${CARD_SERIAL// /}; CARD_UUID=${CARD_UUID// /}; CARD_NAME=${CARD_NAME# }
    bus=${bus// /}; CARD_BDF=$(tr 'A-F' 'a-f' <<< "${bus: -12}")
    CARD_DEVID=$(cat "/sys/bus/pci/devices/$CARD_BDF/device" 2>/dev/null)
    CARD_PORT=$(root_port "$CARD_BDF"); CARD_SLOT="root port $CARD_PORT"
    BLOWER_NAME=${BLOWERS[$CARD_PORT]:-}; BLOWER_TACH=$(blower_tach_for "$CARD_BDF")
    export CARD_IDX CARD_BDF CARD_SERIAL CARD_UUID CARD_NAME CARD_DEVID CARD_PORT CARD_SLOT BLOWER_TACH BLOWER_NAME
}

xid_count() {   # kernel GPU error reports (NVIDIA Xid) for the card, since $1 (journalctl --since) or this boot
    local tag="PCI:${CARD_BDF%.*}"
    if [[ -n ${1:-} ]]; then journalctl -k --since "$1" --no-pager 2>/dev/null | grep -c "Xid ($tag)"
    else journalctl -k -b --no-pager 2>/dev/null | grep -c "Xid ($tag)"; fi
}

line() { printf '%s\n' "$(printf "${1:-─}%.0s" $(seq $W))"; }
# box <title> <right-hand text> "label|text"...: long text wraps under its label
box() {
    local title=$1 right=$2 row lab txt first l; shift 2
    echo; printf "${CYN}"; line ═
    printf " ${B}%-*s${N}${CYN}%s\n" $((W - ${#right} - 2)) "$title" "$right"
    line ═; printf "${N}"
    for row in "$@"; do
        lab=${row%%|*}; txt=${row#*|}; first=1
        while IFS= read -r l; do
            if (( first )); then printf " ${B}%-13s${N} %s\n" "$lab" "$l"; first=0; else printf " %-13s %s\n" "" "$l"; fi
        done < <(fold -s -w $((W - 16)) <<< "$txt")
    done
    printf "${DIM}"; line ─; printf "${N}"
}
pass_banner() { printf "\n ${GRN}██ PASS ██${N}  ${B}%s${N}\n" "$1"; }
fail_banner() { printf "\n ${RED}██ FAIL ██${N}  ${B}%s${N}\n" "$1"; }
warn_banner() { printf "\n ${YEL}██ %s ██${N}  ${B}%s${N}\n" "${2:-NOTE}" "$1"; }
hms() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }
