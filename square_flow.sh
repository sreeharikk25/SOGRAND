#!/bin/bash
# Corrected workflow for the SQUARE code simulation

# Define file names
ORIG_DATA="original_data.bin"
ENCODED_DATA="encoded_square.bin"
CORRUPTED_LLRS="corrupted_llrs_square.bin"
DECODED_DATA="decoded_square.bin"

# Define SNR
SNR_DB="1.0"

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
echo "Encoding data..."
./square_encoder ${ORIG_DATA} ${ENCODED_DATA}

# 4. Simulate the AWGN channel and generate LLRs
#    The '2' specifies the code dimension for rate calculation
echo "Simulating channel..."
./channel_sim ${ENCODED_DATA} ${CORRUPTED_LLRS} ${SNR_DB} 2

# 5. Decode the LLRs to recover the data
echo "Decoding data..."
./square_decoder ${CORRUPTED_LLRS} ${DECODED_DATA}

# 6. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA}

echo "--- Pipeline finished ---"
