# CMP 170HX test suite

Tests for **NVIDIA CMP 170HX** cards running the community memory unlock (64 GB on the 8 GB board
`10de:20c2`, 40 GB on the 10 GB board `10de:2082`). Use them to check a card before you keep it, or to
document a faulty one for the seller.

The suite came out of diagnosing a card that **corrupted data in its HBM under load** while passing light
tests, most memtest_vulkan runs, and every compute test. The tests here catch that kind of fault in
seconds, show what they are doing on screen, and save their raw output, so a test session can be filmed
or shared as evidence.

## The tests

| Command | Time | What it checks | Pass rule |
|---|---|---|---|
| `cmp-identity` | instant | Serial, UUID, VBIOS, memory size, PCIe link, kernel GPU errors | CMP 170HX with full unlocked memory, 0 Xid |
| `cmp-hbm-test [min]` | 10 min (5 is plenty) | **HBM step test**: all free VRAM, alternating a light walking-bit pattern with full-bandwidth random data, every word verified | 0 errors |
| `cmp-tensor-test [s]` | 5 min | int8 + fp16 tensor-core matmuls on every SM vs exact reference | 0 errors, every SM exercised |
| `cmp-sm-test` | < 1 min | Fixed workload on every SM, bit-compared with a known-good baseline | bit-identical |
| `cmp-ai-test` | 2–4 min | Real llama.cpp inference on the card + every weight read back from VRAM and compared with the file | expected answer, no NaN, 0 bytes differ |
| `cmp-kernel-check` | instant | Kernel log since the first test today: NVIDIA Xid reports (GSP/PMU crashes, faults) | 0 Xid |
| `cmp-summary` | instant | Table of everything run today on this card | |
| `cmp-hbm-map [min]` | 5 min | **Diagnostic for a failing card**: maps HBM errors to fixed 2 MiB physical blocks | see [below](#can-bad-memory-be-quarantined) |
| `cmp-telemetry` | live | Live panel for a second terminal: temps, power, load, VRAM, clocks, blower rpm, Xid count | |

Every command finds the card itself (the single CMP 170HX in the machine, or `SERIAL=...`), prints a box
explaining what it does, the exact command and the pass rule, shows the tool's live output, and ends with a
PASS / FAIL banner. GPU tests run under `scripts/guard.sh`, which stops them at 72 °C core / 85 °C HBM
(override with `GPU_LIMIT` / `MEM_LIMIT`). Ctrl+C stops a test cleanly and is recorded as STOPPED.
Raw output of every test goes to `runs/<serial>-<date>/`, per-second telemetry to `logs/`.

## Requirements

- Linux with an NVIDIA driver that runs the card unlocked, and the CUDA toolkit (`nvcc`, compute capability `sm_80`)
- Read access to the kernel log (`journalctl -k`); on most distros the desktop user has it
- For `cmp-ai-test`: llama.cpp built with CUDA (`cmake -B build -DGGML_CUDA=ON && cmake --build build -j`)
  and a GGUF model that fits on one card. Development used Qwen3.8-27B Q8_0 (29 GB).

Developed on driver 610.57.04 (open kernel modules with an unlock patch set), CUDA 12.8, Linux 7.0, a
Threadripper X399 board. Other setups should work; reports welcome.

## Install

```bash
git clone https://github.com/dev-boz/cmp170hx-testsuite && cd cmp170hx-testsuite
LLAMA_DIR=~/llama.cpp ./install.sh      # builds bin/, links cmp-* into ~/.local/bin (omit LLAMA_DIR to skip the AI test)
cp config.example.sh config.sh          # optional: model path, blower tach mapping, output folders
```

`./install.sh remove` removes the links. Everything else stays inside the repo folder.

## Testing a card

Test **one card at a time**, alone in the machine if you can. Before starting:

- Cap the power: `sudo nvidia-smi -i <index> -pl 180` (resets every boot). These boards run hot; 180 W
  costs little performance.
- Make sure the blower works. Setting it to full speed in the BIOS for the test is simplest. If its tach is
  on a motherboard header, map it in `config.sh` so guard.sh can stop a test on a stall.

Suggested order (about 40 minutes):

1. `cmp-identity` (run it first: it starts the day's log window for `cmp-kernel-check`)
2. **memtest_vulkan** ([GpuZelenograd/memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan)),
   its standard 5-minute test. Run it early: right after a cold boot is when the bad card failed it.
   Press Ctrl+C after the verdict, or it keeps running and holds the VRAM.
3. **gpu-burn** ([wilicc/gpu-burn](https://github.com/wilicc/gpu-burn)):
   `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<index> ./gpu_burn -tc -m 95% 300`.
   It lists every GPU at start-up; `Initialized device 0 with ~64910 MB` shows it picked the CMP.
   Pass = `errors: 0` on every line and `GPU 0: OK`.
4. `cmp-hbm-test 5`, `cmp-tensor-test`, `cmp-sm-test`, `cmp-ai-test`
5. `cmp-kernel-check`, `cmp-summary`

`docs/filming.md` has a run sheet for recording the whole thing in one take.

## What a bad card looks like

From a card that failed (unlocked 64 GB, 180 W, normal temperatures, faulty in every slot and on every
power cable, and alone in the system):

- **Error signature:** whole **128-byte blocks** (one HBM burst) read back as zeros or unrelated data. Not
  single flipped bits.
- **Step test:** errors from the **first second** of every run. Error counts vary enormously from run to run
  (128 to 670,000 bad words per 10 minutes), so judge pass/fail, not counts.
- **gpu-burn:** 0.6–1.9 *billion* errors on the first status line, verdict `FAULTY`.
- **memtest_vulkan:** unreliable on this fault: failed right after a cold boot, passed most other runs.
- **Light tests pass** (upload VRAM, read it back slowly). Sustained full-bandwidth traffic is needed.
- **Firmware crashes under load:** GSP crash (`Xid 119`, `Xid 1 GSP task exception`, `Xid 154`) or PMU halt
  (`Xid 62`). `nvidia-smi` then shows `[GPU requires reset]` or `[N/A]`. The card needs a full power-off
  (PSU switch, ~30 s); a warm reboot is not enough.
- **AI inference:** all output scores NaN, or corrupted weights in VRAM.
- Compute (tensor cores, SM check) was clean: the fault was in the memory, not the cores.

## Can bad memory be quarantined?

Some owners fix a card by keeping a few bad memory blocks allocated so nothing else can use them. That
works when the errors sit at a few **fixed physical locations** (typically repeating single-bit flips at the
same addresses). `cmp-hbm-map` tells you which case you have: it holds the same 2 MiB physical blocks for
the whole run and logs which ones fail and how often.

On the bad card above, 7,652 of 31,774 blocks (**24% of VRAM**) failed within 58 s. Covering 99% of the
errors would have taken 11.6 GB, and the errors **followed timing, not location**: 85% landed in what the
first thread blocks of each GPU launch were touching in the first few milliseconds, right after the jump
from idle to full memory traffic. The blocks that "repeated" were just the ones always touched at launch
start. That is a load-transient fault (power delivery, PHY or clock margin), not weak cells, and quarantine
cannot fix it. The run also crashed the card's firmware.

## Notes

- **PCIe x4:** stock boards lack the AC-coupling capacitors on 12 of the 16 PCIe lanes, so they train at x4.
  The community "cap mod" fits them for x16. It only affects transfer speed (model loading, multi-GPU); the
  memory tests are unaffected. If you might return the card, test before soldering.
- **Host RAM:** on machines with non-ECC RAM a bad cell can corrupt files in the page cache. `cmp-ai-test`
  first compares the model through the page cache with a direct disk read, and the weight comparison reads
  the file with O_DIRECT, so a host-RAM fault is not blamed on the card.
- **sm-verify baseline:** `baselines/sm-verify-ga100.bin` was recorded on a known-good CMP 170HX (driver
  610.57.04, CUDA 12.8). If `cmp-sm-test` fails on every card including known-good ones (a different
  compiler can change the FP32 results), record your own on a good card:
  `bin/sm-verify record baselines/my-baseline.bin` and set `CMP_SM_BASELINE`.
- `hbm-map` self-test: `HBM_MAP_INJECT=<handle> bin/hbm-map 0.1` corrupts one known block per round, to
  check that detection works.

## License

MIT, see [LICENSE](LICENSE). memtest_vulkan and gpu-burn are separate projects under their own licenses and
are not included.
