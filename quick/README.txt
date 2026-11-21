CMP 170HX QUICK CHECK
=====================

A short, self-contained check of an NVIDIA CMP 170HX you are about to buy (or have just received), on
any Linux machine, yours or the seller's. One file, no install, no sudo, no internet.

    bash cmp-quick.run            about 6 minutes
    bash cmp-quick.run --id       5 seconds, no load: is it the right card, is it unlocked?
    bash cmp-quick.run --vulkan   adds memtest_vulkan, about 11 minutes
    bash cmp-quick.run --long     everything for longer, about 23 minutes
    bash cmp-telemetry.sh         live temps, power and fans: run it in a second terminal
    --serial N                    test the card with this serial, when the machine holds several

Needs NVIDIA driver 570 or newer. Results save to cmp-quick-results/ next to the file, or to your home
folder if that folder is read-only; the screen says where.


THE RESULT
----------
  PASS           No faults found.
  FAIL           Don't pay for it as working.
  INCONCLUSIVE   Not proven good. Read the reason it gives. Usually fixable on the spot:
                   too hot      -> more airflow, then: sudo nvidia-smi -i <id> -pl 180, run again
                   8 GB only    -> that machine has no unlock. See below.
                   card in use  -> stop whatever is using the card (miner, model server), run again
                   power limit cut -> something on that machine lowered the card's power limit
                                  mid-test, often a fan or thermal watcher. Stop or fix it and run
                                  again; don't trust that run.

Before you pay: the serial on the screen must match the sticker on the card you take away.


ON THE SELLER'S MACHINE
-----------------------
1. Ask beforehand: "Is it running Linux with the unlock, so nvidia-smi shows 65536 MiB?"
2. Open a terminal in the kit's folder and run: bash cmp-quick.run --id
3. It shows 65536 MiB: run: bash cmp-quick.run
   Several cards in the machine: it asks which. Check the sticker matches the one you picked.

If it shows 8 GB, or the machine runs Windows, the 64 GB can't be proven there. Your options: test in
your own machine, or get a refund agreement in writing.
   Windows only, to see the card works at 8 GB: extras\memtest_vulkan-v0.5.0.exe
   (pick the CMP entry, wait for "Standard 5-minute test PASSed", then Ctrl+C.)
   That says nothing about the memory above 8 GB.


IN YOUR OWN MACHINE
-------------------
1. Power off at the PSU and wait 30 s. Fit the card, connect both 8-pins, and make sure the blower's
   air reaches it.
2. Boot, then run the kit. To skip cards you already own, list their serials, one per line, in
   ~/.config/cmp-quick/owned-serials; with a new card present it then picks that one automatically.
3. Not showing 65536 MiB: the memory unlock isn't active on this machine; re-run it for your setup.


WHAT IT TESTS
-------------
  HBM step test (3 min)   Fills all free VRAM, light pattern then full-speed random data, checks
                          every word. A known-bad card failed this within about 1 second, in all 3
                          runs of this pattern.
  Tensor cores (45 s)     int8 and fp16 maths on every SM against the exact answer.
  SM bit-exact (< 1 min)  Fixed calculation compared with a known-good card.
  gpu-burn (60 s)         Independent stress test. Its libraries are in extras/.
  AI test (seconds)       Small model on real llama.cpp: answers Paris, no NaN, weights read back
                          intact. Checks about 0.4 GB only; the HBM test is the memory check.
  memtest_vulkan (5 min)  Only with --vulkan or --long. Independent, and it runs first: the known-bad
                          card once failed it straight after a cold boot, but passed most runs.
                          If the machine has no Vulkan it shows SKIPPED and the rest carries on.
  Kernel log              Xid errors before and during the test, if the user can read the log.

It stops at the first failure. It also stops at 72C core or 85C HBM (that gives INCONCLUSIVE, not FAIL).


WHAT A PASS DOESN'T COVER
-------------------------
  - It was developed against ONE known-bad card. Other faults may need longer: use --long.
  - That card failed memtest_vulkan only straight after a cold boot, so if the machine has been on for
    hours, ask for a full power-off first if the owner doesn't mind.
  - The AI test uses a tiny model. Run a big one once you get the card home.
  - ECC and row-remap history reads N/A on the unlocked driver, so past memory errors can't be seen.
  - Whether the blower is spinning. Look at it.


BUILDING THE KIT
----------------
The repository holds source only; the kit contains third-party binaries (glibc and libstdc++ copied from
your machine, cuBLAS, gpu-burn, memtest_vulkan, a llama.cpp build and a model), so you build it yourself:

  1. make                                          (in the repository root: the CUDA test programs)
     make weight-readback LLAMA_DIR=<llama.cpp>    (llama.cpp built with -DGGML_CUDA=ON)
  2. gpu-burn: build it (github.com/wilicc/gpu-burn) and put memtest_vulkan next to it:
       OFFSHELF/gpu-burn/{gpu_burn,compare.cu}  OFFSHELF/memtest_vulkan  (+ optional the Windows .exe)
  3. OFFSHELF=... LLAMA_LIB=<llama.cpp>/build/bin MODEL=<small .gguf> quick/build.sh [out-folder]
     This writes cmp-quick.run, extras/ and this README to the out-folder (default ~/cmp-quick-kit).
     Needs a Debian/Ubuntu-style machine (it checks the copied libraries with dpkg) and nvcc.
  4. Copy the out-folder to a USB stick. Test the build first: quick/test.sh runs the verdict logic
     against fake hardware, no GPU needed.
