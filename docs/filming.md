# Filming a test session in one take

A continuous video from sealed box to final summary is strong evidence, whether the card passes or fails.

## Before recording
- Terminal font large enough for the camera, window at least ~120 columns. Two panes: tests on top,
  `cmp-telemetry` below (it shows a clock, elapsed time, temperatures, power, load and Xid count, which
  proves the card is really loaded and the take is continuous).
- Blower to full speed in the BIOS, or check it ramps.
- Know which slot the card goes in and that its blower and power cables reach.

## On camera
1. Unbox. Show the card all round and the serial-number sticker up close.
2. Power off at the PSU, install the card, connect power, boot.
3. `nvidia-smi` - the driver sees the card.
4. `sudo nvidia-smi -i <index> -pl 180`
5. `cmp-telemetry` in the second pane; leave it running.
6. `cmp-identity` - hold the sticker next to the screen so the serials can be compared.
7. memtest_vulkan standard 5-minute test (pick the CMP 170HX entry), Ctrl+C after the verdict.
8. gpu-burn, 300 s (see the README for the exact command).
9. `cmp-hbm-test 5`, `cmp-tensor-test`, `cmp-sm-test`, `cmp-ai-test`
10. `cmp-kernel-check`, then `cmp-summary`.

## If something goes wrong on camera
- A test that will not start because VRAM is full: another tester (often memtest_vulkan) is still running.
  Its terminal needs a Ctrl+C.
- The card crashes (`nvidia-smi` shows `[N/A]` or `GPU requires reset`): run `cmp-kernel-check` to show
  the Xid errors on screen, then power off fully. Do not keep testing a crashed card.
- Stopped a test by mistake: it is recorded as STOPPED, not FAIL; just run it again.
