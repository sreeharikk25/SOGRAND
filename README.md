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
| `Tmax`    | `UINT64_MAX` (tunable via `TMAX_PER_COMPONENT`) | max guesses per component decode (see *Performance notes*) |
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

`cubic_decoder1.cu` has had the following throughput fixes applied:

- **Accurate timing.** The *Throughput: … blocks/sec* figure now comes from
  `cudaEvent` timers around the kernel launch(es), i.e. real device wall-clock.
  It previously used `clock()`, which measures host CPU time and under-counts
  while the host is blocked in `cudaDeviceSynchronize()` — inflating the number.
- **No idle threads.** `THREADS_PER_BLOCK` is now `256` (= `n*n`). The component
  decode only ever runs for `tid < n*n`, so the previous value of `512` left
  half the block idle through the most expensive phase.
- **Tunable abandonment cap.** `Tmax` is exposed as the `TMAX_PER_COMPONENT`
  macro. It defaults to `UINT64_MAX` (exact C-reference behavior). Setting a
  finite value (e.g. `-DTMAX_PER_COMPONENT=100000`) caps worst-case per-line
  enumeration, which reduces warp divergence and improves throughput **at the
  cost of changed decoded output / BER** — benchmark BER before relying on it.

Remaining known limiters (not yet changed, since they need restructuring and
GPU profiling to do safely):

- **Low occupancy from large per-thread local memory.** Each active thread holds
  several 16-element `double` vectors plus a 512-byte workspace, and the SOGRAND
  helper allocates further per-thread arrays — over ~2 KB/thread, which spills
  registers to local memory (note `cudaLimitStackSize` is raised to 16 KB).
- **Convergence checked after every one of the 3 phases per iteration** (up to
  ~90 full-cube re-encodes with barriers per codeword).

Profile with `nvcc -Xptxas -v` (register/local-memory usage) and Nsight Compute
(achieved occupancy) before and after any further changes. For reference,
`ptxas -v` reports a **3744-byte stack frame and 100 registers per thread** for
the decode kernel on `sm_89` — the local-memory pressure that caps occupancy.

### Measured example (NVIDIA RTX 4070 Laptop GPU, `sm_89`, CUDA 12.0)

1000 codewords, `corrupted_llrs_cubic.bin` @ SNR 2.0 dB, all runs **BER = 0**:

| `Tmax` (`TMAX_PER_COMPONENT`) | GPU time | Throughput | BER |
|-------------------------------|----------|------------|-----|
| `UINT64_MAX` (default) | 75.8 s | 13.19 blocks/s | 0 |
| `100000` | 75.8 s | 13.19 blocks/s | 0 |
| `1000` | 68.4 s | 14.61 blocks/s | 0 |

Note that capping `Tmax` at 100000 has **no effect**: the average is only
~526 guesses per component decode (≈1.96 M guesses/block over ≈3725 component
decodes), so a generous cap never binds. Only a tight cap (≤1000) helps, and
only by ~11% here. The dominant cost is the sheer volume of serial enumeration
plus the low occupancy above — not the abandonment cap.
