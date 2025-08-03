#!/bin/bash
# Dual encoding/decoding workflow for CUBIC code simulation using AWGN channel

# Define file names
ORIG_DATA="original_data.bin"
FIRST_ENCODED="first_encoded_cubic.bin"
SECOND_ENCODED="second_encoded_cubic.bin"
CORRUPTED_BITS="corrupted_bits_cubic.bin"
CORRUPTED_LLRS="corrupted_llrs_cubic.bin"
FIRST_DECODED="first_decoded_cubic.bin"
FINAL_DECODED="final_decoded_cubic.bin"

# Define SNR in dB for the channel
SNR_DB="2.0"

echo "--- Starting DUAL CUBIC Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"

# 1. Compile all necessary C programs
echo "Compiling programs..."
gcc generate_bin.c -o generate_bin
gcc cubic_encoder.c -o cubic_encoder -lm
gcc channel_sim.c -o channel_sim -lm
gcc llr_calculator.c -o llr_calculator -lm
gcc cubic_decoder.c -o cubic_decoder -lm
gcc comparator.c -o comparator

# 2. Generate a random binary file
echo "Generating random data..."
./generate_bin

# 3. First encoding: original data -> first encoded
echo "First encoding..."
./cubic_encoder ${ORIG_DATA} ${FIRST_ENCODED}

# 4. Second encoding: first encoded -> second encoded (dual encoding)
#echo "Second encoding (dual encoding)..."
#./cubic_encoder ${FIRST_ENCODED} ${SECOND_ENCODED}

# 5. Simulate the AWGN channel (outputs corrupted bits)
echo "Simulating channel (generating corrupted bits)..."
./channel_sim ${FIRST_ENCODED} ${CORRUPTED_BITS} ${SNR_DB}

# 6. Calculate LLRs from the corrupted bits
echo "Calculating LLRs from corrupted bits..."
./llr_calculator ${CORRUPTED_BITS} ${CORRUPTED_LLRS} ${SNR_DB} 3

# 7. First decoding: LLRs -> first decoded data
echo "First decoding..."
./cubic_decoder ${CORRUPTED_LLRS} ${FIRST_DECODED}

# 8. Calculate LLRs from first decoded data for second decoding
# For the second decoding, we need to convert the decoded bits back to LLRs
# We'll use perfect channel assumption for the second stage
#echo "Preparing for second decoding (converting bits to LLRs)..."
# Create a temporary file with very high SNR LLRs (near-perfect channel)
#HIGH_SNR="20.0"  # Very high SNR for near-perfect reconstruction
#./llr_calculator ${FIRST_DECODED} ${CORRUPTED_LLRS} ${HIGH_SNR} 3

# 9. Second decoding: second LLRs -> final decoded data
#echo "Second decoding (dual decoding)..."
#./cubic_decoder ${CORRUPTED_LLRS} ${FINAL_DECODED}

# 10. Compare the original and final decoded files
echo "Comparing original vs final decoded results..."
./comparator ${ORIG_DATA} ${FIRST_DECODED}

# Additional comparisons for debugging
#echo ""
#echo "--- Additional Comparisons for Analysis ---"
#echo "Comparing original vs first encoded:"
#./comparator ${ORIG_DATA} ${FIRST_ENCODED}

#echo ""
#echo "Comparing first encoded vs second encoded:"
#./comparator ${FIRST_ENCODED} ${SECOND_ENCODED}

#echo ""
#echo "Comparing original vs first decoded:"
#./comparator ${ORIG_DATA} ${FIRST_DECODED}

echo ""
echo "--- File Sizes (for verification) ---"
echo "Original data size: $(wc -c < ${ORIG_DATA}) bytes"
echo "First encoded size: $(wc -c < ${FIRST_ENCODED}) bytes"
#echo "Second encoded size: $(wc -c < ${SECOND_ENCODED}) bytes"
echo "First decoded size: $(wc -c < ${FIRST_DECODED}) bytes"
#echo "Final decoded size: $(wc -c < ${FINAL_DECODED}) bytes"

echo "--- Dual Pipeline finished ---"
