#!/bin/bash
# Final, corrected workflow for the CUBIC code simulation using an AWGN channel.

# Define file names
ORIG_DATA="original_data.bin"
ENCODED_DATA="encoded_cubic.bin"
CORRUPTED_LLRS="corrupted_llrs_cubic.bin"
DECODED_DATA="decoded_cubic.bin"

# Define SNR in dB for the channel
SNR_DB="2.0"

echo "--- Starting CUBIC Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"

# 1. Compile all necessary C programs
echo "Compiling programs..."
gcc generate_bin.c -o generate_bin
gcc cubic_encoder.c -o cubic_encoder -lm
gcc channel_sim.c -o channel_sim -lm
gcc cubic_decoder.c -o cubic_decoder -lm
gcc comparator.c -o comparator
# Added by Sarah 
gcc cubic_decoder_updated.c -o cubic_decoder_updated -lm 
gcc cubic_decoder_final.c -o cubic_decoder_final -lm

# 2. Generate a random binary file
./generate_bin

# 3. Encode the data using the cubic encoder
echo "Encoding data..."
# ./cubic_encoder ${ORIG_DATA} ${ENCODED_DATA}
./cubic_encoder ${ORIG_DATA} ${ENCODED_DATA}
# 4. Simulate the AWGN channel and generate LLRs
#    The '3' specifies the cubic code dimension for rate calculation.
echo "Simulating channel..."
# ./channel_sim ${ENCODED_DATA} ${CORRUPTED_LLRS} ${SNR_DB} 3
./channel_sim ${ENCODED_DATA} ${CORRUPTED_LLRS} ${SNR_DB} 3

# 5. Decode the LLRs to recover the data
#    The decoder no longer needs the SNR value.
echo "Decoding data..."
# ./cubic_decoder ${CORRUPTED_LLRS} ${DECODED_DATA}
# ./cubic_decoder_updated ${CORRUPTED_LLRS} ${DECODED_DATA}
./cubic_decoder_final ${CORRUPTED_LLRS} ${DECODED_DATA}

# 6. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA}

echo "--- Pipeline finished ---"
