#!/bin/bash
# CUDA Cubic Code Simulation Pipeline - Exact match to your original workflow
# This script uses CUDA-accelerated encoding and decoding

# Define file names (same as original)
ORIG_DATA="original_data.bin"
ENCODED_DATA="encoded_cubic.bin"
CORRUPTED_LLRS="corrupted_llrs_cubic.bin"
DECODED_DATA="decoded_cubic.bin"

# Define SNR in dB for the channel
SNR_DB="2.0"

echo "--- Starting CUDA CUBIC Code Simulation Pipeline (SNR = ${SNR_DB} dB) ---"

# Check CUDA availability
check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        echo "ERROR: NVCC not found. Please install NVIDIA CUDA Toolkit."
        echo "Run: sudo apt install nvidia-cuda-toolkit"
        exit 1
    fi

    if ! nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi not found. GPU may not be available."
        echo "CUDA programs will fall back to CPU if needed."
    else
        echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    fi
}

# 1. Check CUDA installation
echo "Checking CUDA installation..."
check_cuda

# 2. Compile all necessary programs (5 total - matching your request)
echo "Compiling programs..."
make clean > /dev/null 2>&1

echo "  - Compiling data generator..."
gcc -O3 -std=c99 generate_bin.c -o generate_bin

echo "  - Compiling CUDA cubic encoder..."
nvcc -O3 -arch=sm_60 -std=c++11 cuda_cubic_encoder.cu -o cuda_cubic_encoder

echo "  - Compiling channel simulator..."
gcc -O3 -std=c99 channel_sim.c -o channel_sim -lm

echo "  - Compiling CUDA cubic decoder..."
nvcc -O3 -arch=sm_60 -std=c++11 cuda_cubic_decoder.cu -o cuda_cubic_decoder

echo "  - Compiling comparator..."
gcc -O3 -std=c99 comparator.c -o comparator

# Check if all executables were created
if [[ ! -f generate_bin || ! -f cuda_cubic_encoder || ! -f channel_sim || ! -f cuda_cubic_decoder || ! -f comparator ]]; then
    echo "ERROR: Failed to compile one or more programs."
    echo "Please check CUDA installation and try again."
    exit 1
fi

echo "All programs compiled successfully!"

# 3. Generate a random binary file
echo "Generating random data..."
./generate_bin
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to generate random data."
    exit 1
fi

# 4. Encode the data using the CUDA cubic encoder
echo "CUDA Encoding data..."
./cuda_cubic_encoder ${ORIG_DATA} ${ENCODED_DATA}
if [[ $? -ne 0 ]]; then
    echo "ERROR: CUDA encoding failed."
    exit 1
fi

# 5. Simulate the AWGN channel and generate LLRs
#    The '3' specifies the cubic code dimension for rate calculation.
echo "Simulating AWGN channel..."
./channel_sim ${ENCODED_DATA} ${CORRUPTED_LLRS} ${SNR_DB} 3
if [[ $? -ne 0 ]]; then
    echo "ERROR: Channel simulation failed."
    exit 1
fi

# 6. Decode the LLRs to recover the data using CUDA decoder
echo "CUDA Decoding data..."
./cuda_cubic_decoder ${CORRUPTED_LLRS} ${DECODED_DATA}
if [[ $? -ne 0 ]]; then
    echo "ERROR: CUDA decoding failed."
    exit 1
fi

# 7. Compare the original and decoded files
echo "Comparing results..."
./comparator ${ORIG_DATA} ${DECODED_DATA}
comparison_result=$?

# 8. Display pipeline summary
echo ""
echo "--- CUDA Pipeline Summary ---"
echo "Original file:    ${ORIG_DATA}"
echo "Encoded file:     ${ENCODED_DATA}"
echo "Channel LLRs:     ${CORRUPTED_LLRS}"
echo "Decoded file:     ${DECODED_DATA}"
echo "SNR:              ${SNR_DB} dB"
echo "Code:             Cubic (15,10) CRC"
echo "Encoder:          CUDA Accelerated"
echo "Decoder:          CUDA Accelerated"

# Check file sizes for verification
if [[ -f ${ORIG_DATA} && -f ${ENCODED_DATA} && -f ${DECODED_DATA} ]]; then
    orig_size=$(stat -c%s ${ORIG_DATA})
    encoded_size=$(stat -c%s ${ENCODED_DATA})
    decoded_size=$(stat -c%s ${DECODED_DATA})

    echo "File sizes:"
    echo "  Original:  ${orig_size} bytes"
    echo "  Encoded:   ${encoded_size} bytes"
    echo "  Decoded:   ${decoded_size} bytes"

    # Calculate code rate
    rate=$(echo "scale=4; ${orig_size}/${encoded_size}" | bc -l 2>/dev/null || echo "~0.296")
    echo "  Code rate: ${rate}"
fi

if [[ $comparison_result -eq 0 ]]; then
    echo "STATUS: SUCCESS - Decoding completed successfully!"
else
    echo "STATUS: FAILURE - Errors detected in decoding."
fi

echo "--- Pipeline finished ---"

# Optional: Run additional tests if requested
if [[ "$1" == "--extended" ]]; then
    echo ""
    echo "--- Running Extended Tests ---"

    # Test multiple SNR values
    echo "Testing multiple SNR values..."
    for snr in 1.0 1.5 2.5 3.0; do
        echo "  Testing SNR = ${snr} dB"
        ./channel_sim ${ENCODED_DATA} "llr_${snr}.bin" ${snr} 3
        ./cuda_cubic_decoder "llr_${snr}.bin" "decoded_${snr}.bin"
        echo -n "    Result: "
        ./comparator ${ORIG_DATA} "decoded_${snr}.bin" | grep -E "(SUCCESS|FAILURE)"
    done

    # Performance test with larger data
    echo "Performance test with larger dataset..."
    dd if=/dev/urandom of=large_data.bin bs=6250 count=1 2>/dev/null
    echo "  Encoding..."
    time ./cuda_cubic_encoder large_data.bin encoded_large.bin
    echo "  Channel..."
    time ./channel_sim encoded_large.bin llr_large.bin 1.5 3
    echo "  Decoding..."
    time ./cuda_cubic_decoder llr_large.bin decoded_large.bin
    echo "  Verification..."
    ./comparator large_data.bin decoded_large.bin

    # Cleanup extended test files
    rm -f llr_*.bin decoded_*.bin large_data.bin encoded_large.bin llr_large.bin decoded_large.bin

    echo "--- Extended Tests Complete ---"
fi

# Optional: Cleanup temporary files
if [[ "$2" == "--clean" ]]; then
    echo "Cleaning temporary files..."
    rm -f ${ORIG_DATA} ${ENCODED_DATA} ${CORRUPTED_LLRS} ${DECODED_DATA}
    echo "Cleanup complete."
fi

# Show usage information
if [[ "$1" == "--help" ]]; then
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --extended    Run additional tests with multiple SNR values"
    echo "  --clean       Clean temporary files after completion"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Basic pipeline"
    echo "  $0 --extended         # With extended tests"
    echo "  $0 --extended --clean # Extended tests + cleanup"
    echo ""
    echo "Files generated:"
    echo "  ${ORIG_DATA}          # Original random data"
    echo "  ${ENCODED_DATA}       # CUDA encoded codeword"
    echo "  ${CORRUPTED_LLRS}     # Channel LLRs"
    echo "  ${DECODED_DATA}       # CUDA decoded result"
fi

exit $comparison_result
