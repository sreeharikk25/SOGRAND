#!/bin/bash
# Corrected workflow for the SQUARE code simulation

# Define file names
ORIG_DATA="original_data.bin"
ENCODED_DATA_STAGE1="encoded1_square.bin"
# ENCODED_DATA_STAGE2="encoded2_square.bin"
CORRUPTED_LLRS="corrupted_llrs_square.bin"
DECODED_DATA_STAGE1="decoded1_square.bin"
DECODED_DATA_STAGE2="decoded2_square.bin"


# Define SNR
SNR_DB="2.0"

echo "--- Starting SQUARE Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"



# 1. Compile all necessary C programs
echo "Compiling programs..."
gcc generate_bin.c -o generate_bin
gcc square_encoder.c -o square_encoder -lm
gcc channel_sim.c -o channel_sim -lm
gcc square_decoder.c -o square_decoder -lm
gcc comparator.c -o comparator

# 2. Generate a random binary file
./generate_bin

# 3. Encode the data using the square encoder
echo "Encoding data stage 1..."
./square_encoder ${ORIG_DATA} ${ENCODED_DATA_STAGE1}
# echo "Encoding data stage 2..."
# ./square_encoder ${ENCODED_DATA_STAGE1} ${ENCODED_DATA_STAGE2}

# 4. Simulate the AWGN channel and generate LLRs
#    The '2' specifies the code dimension for rate calculation
echo "Simulating channel..."
./channel_sim ${ENCODED_DATA_STAGE1} ${CORRUPTED_LLRS} ${SNR_DB} 2

# 5. Decode the LLRs to recover the data
echo "Decoding data stage 1..."
./square_decoder ${CORRUPTED_LLRS} ${DECODED_DATA_STAGE1}
# echo "Decoding data stage 2..."
# ./square_decoder ${DECODED_DATA_STAGE1} ${DECODED_DATA_STAGE2}

# 6. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA_STAGE1}

echo "--- Pipeline finished ---"
