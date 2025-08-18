#!/bin/bash
# CUDA Cubic Code Simulation Pipeline - Simple Version
# Define file names
ORIG_DATA="original_data.bin"
ENCODED_DATA="encoded_cubic.bin"
CORRUPTED_BITS="corrupted_bits.bin"
CORRUPTED_LLRS="corrupted_llrs.bin"
DECODED_DATA="decoded_cubic.bin"

# Define SNR in dB for the channel
SNR_DB="2.0"

echo "--- Starting CUDA CUBIC Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"

# 1. Compile all necessary programs
echo "Compiling programs..."
gcc generate_bin.c -o generate_bin
nvcc cuda_cubic_encoder.cu -o cuda_cubic_encoder
gcc channel_sim.c -o channel_sim -lm
gcc llr_calculator.c -o llr_calculator -lm
nvcc cuda_cubic_decoder.cu -o cuda_cubic_decoder
gcc comparator.c -o comparator

# 2. Generate a random binary file
echo "Generating random data..."
./generate_bin

# 3. CUDA encoding: original data -> encoded data
echo "CUDA encoding..."
./cuda_cubic_encoder ${ORIG_DATA} ${ENCODED_DATA}

# 4. Simulate the AWGN channel (outputs corrupted bits)
echo "Simulating channel..."
./channel_sim ${ENCODED_DATA} ${CORRUPTED_BITS} ${SNR_DB}

# 5. Calculate LLRs from the corrupted bits
echo "Calculating LLRs..."
./llr_calculator ${CORRUPTED_BITS} ${CORRUPTED_LLRS} ${SNR_DB} 3

# 6. CUDA decoding: LLRs -> decoded data
echo "CUDA decoding..."
./cuda_cubic_decoder ${CORRUPTED_LLRS} ${DECODED_DATA}

# 7. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA}

echo ""
echo "--- File Sizes ---"
echo "Original data size: $(wc -c < ${ORIG_DATA}) bytes"
echo "Encoded data size: $(wc -c < ${ENCODED_DATA}) bytes"
echo "Decoded data size: $(wc -c < ${DECODED_DATA}) bytes"

echo "--- Pipeline finished ---"                        
