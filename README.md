# SOGRAND Turbo Product and Cubic Code Implementation

This repository contains C and CUDA implementations of SOGRAND (Soft-Output Guessing Random Additive Noise Decoding) for turbo product codes and turbo cubic codes.

## Overview

SOGRAND is a universal decoder that can decode any linear block code by guessing noise patterns in order of decreasing likelihood. This implementation includes:

- **SOGRAND soft-output decoder** with list decoding capabilities
- **Turbo product code** (2D) encoding and iterative decoding
- **Turbo cubic code** (3D) encoding and iterative decoding
- Support for **CRC codes** from Koopman's database
- Support for **extended BCH codes** (stub implementation)
- Both **C** and **CUDA** implementations for performance comparison

## Key Features

- **Universal decoding**: Works with any linear block code
- **Soft output**: Provides reliability information for iterative decoding
- **GPU acceleration**: CUDA implementation for parallel decoding
- **Flexible code support**: CRC polynomials from 3-bit to 29-bit redundancy
- **Monte Carlo simulation**: Full encoding, channel, and decoding pipeline

## Directory Structure

```
.
├── SOGRAND_C/          # C implementation
│   ├── square_encoder.c
│   ├── square_decoder.c
│   ├── cubic_encoder.c
│   ├── cubic_decoder.c
│   ├── channel_sim.c
│   ├── crc_polynomials.c
│   ├── bch_codes.c
│   └── Makefile
├── SOGRAND_CUDA/       # CUDA implementation
│   ├── square_encoder.cu
│   ├── square_decoder.cu
│   ├── cubic_encoder.cu
│   ├── cubic_decoder.cu
│   ├── channel_sim.cu
│   ├── crc_polynomials.cu
│   └── Makefile
└── IMPLEMENTATION_VERIFICATION.md
```

## Building

### C Implementation
```bash
cd SOGRAND_C
make clean && make
```

### CUDA Implementation
```bash
cd SOGRAND_CUDA
make clean && make
```

## Usage

### Product Code (2D)
```bash
# Generate test data
./generate_bin

# Encode
./square_encoder original_data.bin encoded.bin

# Add channel noise
./channel_sim encoded.bin corrupted.bin 2.0

# Decode
./square_decoder corrupted.bin decoded.bin

# Compare results
./comparator original_data.bin decoded.bin
```

### Cubic Code (3D)
```bash
# Similar flow but with cubic encoder/decoder
./cubic_encoder original_data.bin encoded.bin
./cubic_decoder corrupted.bin decoded.bin
```

## Implementation Details

The implementations are verified to match the MATLAB reference code, including:

- Complete CRC polynomial database from Koopman
- SOGRAND algorithm with landslide/mountain error pattern generation
- Turbo decoding with extrinsic information exchange
- Early termination based on syndrome checks

See [IMPLEMENTATION_VERIFICATION.md](IMPLEMENTATION_VERIFICATION.md) for detailed verification information.

## References

1. K. R. Duffy, J. Li, M. Médard. "Capacity-Achieving Guessing Random Additive Noise Decoding". IEEE Transactions on Information Theory 65 (2019): 4023-4040.
2. K. R. Duffy, W. An, M. Médard. "Ordered reliability bits guessing random additive noise decoding." IEEE Transactions on Signal Processing 70 (2022): 4528-4542.
3. P. Yuan, M. Medard, K. Galligan, K. R. Duffy. "Soft-output (SO) GRAND and Iterative Decoding to Outperform LDPC Codes". IEEE Transactions on Wireless Communications (2025).

## License

All code is subject to the GRAND Codebase Non-Commercial Academic Research Use License.