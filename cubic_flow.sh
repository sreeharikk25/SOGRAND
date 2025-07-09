#!/bin/bash

# This script demonstrates a typical workflow for using the
# cubic_encoder, channel_sim, cubic_decoder, and comparator programs.
# It generates a random binary file, encodes it, simulates channel
# errors, decodes the result, and then compare the original data against the decode data.

# Define input and output file names
ORIGDAT_BIN="original_data.bin"
ENCODED_BIN="encoded.bin"
CHANNEL_OUT="channel_out.bin"
DECODED_BIN="decoded_3d.bin"

# Define channel error percentage
ERROR_RATE="0.2"

# Define SNR
SNR_DB="1.5"

# Generate a random binary file
./generate_bin ${ORIGDAT_BIN}

# Encode the binary file using cubic code
./encoder ${ORIGDAT_BIN} ${ENCODED_BIN}

# Add errors into the encoded data at a specified error rate
./channel_sim ${ENCODED_BIN} ${CHANNEL_OUT} ${ERROR_RATE}

# Decode the binary file using cubic code to recover the original data
./cubic_decoder ${CHANNEL_OUT} ${DECODED_BIN} ${SNR_DB}

# Compare the original against the decoded data
./comparator ${ORIGDAT_BIN} ${DECODED_BIN}
