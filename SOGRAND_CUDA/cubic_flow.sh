#!/bin/bash
# cuda_cubic_flow.sh - CUDA-accelerated cubic code simulation

# Define file names
ORIG_DATA="original_data.bin"
ENCODED_DATA="encoded_cubic.bin"
CORRUPTED_LLRS="corrupted_llrs_cubic.bin"
DECODED_DATA="decoded_cubic.bin"

# Define SNR
SNR_DB="2.0"

echo "--- Starting CUDA Cubic Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"

# 1. Generate a random binary file
./generate_bin

# 2. Encode the data using the CUDA cubic encoder
echo "Encoding data with CUDA..."
time ./cuda_cubic_encoder ${ORIG_DATA} ${ENCODED_DATA}

# 3. Simulate the AWGN channel with CUDA
echo "Simulating channel with CUDA..."
time ./cuda_channel_sim ${ENCODED_DATA} ${CORRUPTED_LLRS} ${SNR_DB} 3

# 4. Decode the LLRs with CUDA
echo "Decoding data with CUDA..."
time ./cuda_cubic_decoder ${CORRUPTED_LLRS} ${DECODED_DATA}

# 5. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA}

echo "--- CUDA Pipeline finished ---"
