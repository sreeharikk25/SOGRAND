# SOGRAND — Cubic (3D) Product-Code Decoder

Soft-Output Guessing Random Additive Noise Decoding (**SOGRAND**) for a **cubic
(3-dimensional) product code**, with a reference **C (CPU)** implementation and
a **CUDA (GPU)** implementation.

The pipeline generates random data, encodes it with a 3D product code, passes it
through a simulated AWGN channel to produce log-likelihood ratios (LLRs), decodes
the LLRs with SOGRAND, and compares the result against the original data.

> Licensed for **non-commercial academic research use** — see
> [`GRAND Codebase Non-Commercial Academic Research Use License 021722.pdf`](GRAND%20Codebase%20Non-Commercial%20Academic%20Research%20Use%20License%20021722.pdf).

---

## Repository layout

```
SOGRAND/
├── SOGRAND_C/            # Reference CPU implementation (C)
│   ├── generate_bin.c        # random source-data generator
│   ├── cubic_encoder.c       # 3D product-code encoder
│   ├── channel_sim.c         # AWGN channel + LLR generation
│   ├── cubic_decoder*.c      # SOGRAND decoders (variants)
│   ├── comparator.c          # bit-error comparison of input vs. output
│   └── cubic_flow.sh         # end-to-end pipeline driver
│
└── SOGRAND_CUDA/        # GPU implementation (CUDA)
    ├── generate_bin.c        # random source-data generator
    ├── cubic_encoder.c       # 3D product-code encoder (CPU)
    ├── channel_sim.c         # AWGN channel + LLR generation (CPU)
    ├── cubic_decoder1.cu     # GPU SOGRAND decoder  ← built & run by cubic_flow.sh
    ├── cubic_decoder.cu      # earlier multi-stream GPU variant (not built by default)
    ├── comparator.c          # bit-error comparison of input vs. output
    └── cubic_flow.sh         # end-to-end pipeline driver
```

> **Which decoder is canonical?** `SOGRAND_CUDA/cubic_flow.sh` compiles
> **`cubic_decoder1.cu`** (the "highly optimized" single-launch kernel) into the
> executable named `cubic_decoder`. `cubic_decoder.cu` is an earlier
> stream/batch-based variant kept for reference.

## Code parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n`       | 16    | codeword length per dimension |
| `k`       | 8     | message length per dimension |
| Code      | (16,8)³ | cubic product code, rate `(k/n)³ = 1/8` |
| `L`       | 3     | SOGRAND list size |
| `Imax`    | 30    | max product-code iterations |
| `Tmax`    | `UINT64_MAX` | max guesses per component decode (see *Performance notes*) |
| block     | `16³ = 4096` bits/codeword, `8³ = 512` info bits/codeword |

---

## Requirements

- **C toolchain:** `gcc` with `-lm`
- **GPU build:** NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU
- Linux (the driver scripts are bash)

## Build & run

### GPU pipeline

```bash
cd SOGRAND_CUDA
./cubic_flow.sh
```

This compiles every stage, generates data, runs encode → channel → GPU-decode →
compare, and prints decoder statistics (iterations, guesses, and throughput).

To build the GPU decoder on its own:

```bash
nvcc -O3 -arch=native -use_fast_math -Xptxas -v \
     -Xcompiler -march=native -Xcompiler -O3 \
     -o cubic_decoder cubic_decoder1.cu
./cubic_decoder corrupted_llrs_cubic.bin decoded_cubic.bin
```

> `-arch=native` targets **the GPU on the build machine**. To run the binary on a
> different GPU, replace it with the appropriate `-arch=sm_XX` (e.g. `sm_80`,
> `sm_86`, `sm_90`).

### CPU reference pipeline

```bash
cd SOGRAND_C
./cubic_flow.sh
```

## Changing the SNR

Edit the `SNR_DB` variable at the top of the relevant `cubic_flow.sh`:

```sh
SNR_DB="2.5"   # Eb/N0 in dB
```

---

## Performance notes (GPU throughput)

The reported *Throughput: … blocks/sec* is currently derived from `clock()`,
which measures **CPU process time, not wall-clock GPU time**. Because the host
blocks in `cudaDeviceSynchronize()` while the kernel runs, this figure does not
reflect real decoder throughput. Prefer `cudaEvent` timers (or a wall-clock
source such as `clock_gettime(CLOCK_MONOTONIC, …)`) around the kernel launch and
synchronize.

Real GPU throughput of `cubic_decoder1.cu` is also limited by:

- **Low occupancy from large per-thread local memory.** Each active thread holds
  several 16-element `double` vectors plus a 512-byte workspace, and the SOGRAND
  helper allocates further per-thread arrays — over ~2 KB/thread, which spills
  registers to local memory (note `cudaLimitStackSize` is raised to 16 KB).
- **Idle threads during the dominant phase.** The block launches
  `THREADS_PER_BLOCK = 512` threads, but the component decode runs only for
  `tid < n*n = 256`; the other 256 threads idle through the most expensive work.
- **Unbounded `Tmax`.** With `Tmax = UINT64_MAX`, a single hard-to-decode
  component can enumerate for a very long time and stall its entire warp/block
  (lockstep divergence). A finite `Tmax` caps worst-case latency.
- **Convergence checked after every one of the 3 phases per iteration** (up to
  ~90 full-cube re-encodes with barriers per codeword).

See the "Performance notes" section above for suggested directions before
benchmarking.
