#!/bin/bash
# cuda_square_flow.sh - CUDA-accelerated square code simulation

# Define file names
ORIG_DATA="original_data.bin"
ENCODED_DATA="encoded_square.bin"
CORRUPTED_LLRS="corrupted_llrs_square.bin"
DECODED_DATA="decoded_square.bin"

# Define SNR
SNR_DB="2.0"

echo "--- Starting CUDA Square Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"

# 1. Generate a random binary file
./generate_bin

# 2. Encode the data using the CUDA square encoder
echo "Encoding data with CUDA..."
time ./square_encoder ${ORIG_DATA} ${ENCODED_DATA}

# 3. Simulate the AWGN channel with CUDA
echo "Simulating channel with CUDA..."
time ./channel_sim ${ENCODED_DATA} ${CORRUPTED_LLRS} ${SNR_DB} 2

# 4. Decode the LLRs with CUDA
echo "Decoding data with CUDA..."
time ./square_decoder ${CORRUPTED_LLRS} ${DECODED_DATA}

# 5. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA}

echo "--- CUDA Pipeline finished ---"


