#!/usr/bin/env python3
"""Host-RAM check before the AI test (CPU only, does not touch the GPU).
This PC's non-ECC system RAM has one known stuck bit. llama.cpp loads the model through the Linux page
cache, so a page landing on that bad RAM cell would give a 1-byte model difference that is NOT the GPU's
fault. This reads the model twice: through the page cache (which also preloads it for the AI test) and
with O_DIRECT straight from disk, and compares every byte. Any cached range that differs is evicted
and re-read until it matches.
   model-cache-check.py <model file>"""
import mmap, os, sys, time

path = sys.argv[1]; CH = 64 << 20
fd_d = os.open(path, os.O_RDONLY | os.O_DIRECT); fd_b = os.open(path, os.O_RDONLY)
buf = mmap.mmap(-1, CH); size = os.fstat(fd_b).st_size
t0 = time.time(); fixed = 0; bad_left = 0
print(f"host RAM check of {os.path.basename(path)} ({size / 1e9:.1f} GB): page cache vs direct disk read", flush=True)
for off in range(0, size, CH):
    n = os.preadv(fd_d, [buf], off); d = buf[:n]
    for attempt in range(6):
        b = os.pread(fd_b, n, off)
        if b == d: break
        diffs = [i for i in range(n) if b[i] != d[i]][:4]
        print("  cached copy differs at " + ", ".join(f"file@{off + i} (disk {d[i]:02x}, RAM {b[i]:02x})" for i in diffs)
              + " -> evicting and re-reading", flush=True)
        os.posix_fadvise(fd_b, off, n, os.POSIX_FADV_DONTNEED)
    else:
        bad_left += 1; continue
    if attempt: fixed += 1
print(f"compared {size / 1e9:.1f} GB in {time.time() - t0:.0f} s: "
      + ("RESULT: CLEAN (model in RAM matches disk" + (f"; {fixed} range(s) re-read after a host-RAM error" if fixed else "") + ")"
         if not bad_left else f"RESULT: HOST RAM ERROR in {bad_left} range(s) that would not clear"), flush=True)
sys.exit(1 if bad_left else 0)
