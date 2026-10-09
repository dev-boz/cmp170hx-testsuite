# Optional machine config. Copy to config.sh (git-ignored) and edit.
#
# CMP 170HX boards are passive; most people strap a blower to them. If the blower's tach wire goes to a
# motherboard fan header, map the card's PCIe ROOT PORT to that header so that guard.sh stops a test when
# the blower stalls (< BLOWER_MIN rpm for 3 s) and cmp-telemetry shows its rpm.
#
# Root port of a card:   basename "$(dirname "$(readlink -f /sys/bus/pci/devices/0000:42:00.0)")"
# Fan sensors:           grep . /sys/class/hwmon/hwmon*/name ; cat /sys/class/hwmon/hwmonN/fan*_input
# Value format:          "<hwmon chip-name prefix>:<fanN>"  (hwmonN numbers change between boots; names don't)
#
# BLOWERS["0000:40:03.1"]="it8686:fan2"
# BLOWERS["0000:00:03.1"]="it8792:fan3"
#
# Where logs and test output go (defaults: logs/ and runs/ inside the repo):
# CMP_LOG_DIR=~/cmp-logs
# CMP_RUNS_DIR=~/cmp-runs
#
# Model for cmp-ai-test (any GGUF that fits on one card; Qwen3.8-27B Q8_0 ~29 GB was used in development):
# CMP_MODEL=~/models/Qwen3.8-27B-Q8_0.gguf
# CMP_EXPECT=" Paris"     # expected top token for "The capital of France is"
