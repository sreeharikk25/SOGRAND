# Makefile for CUDA Cubic Decoder - 5 separate programs
# Requires NVIDIA CUDA Toolkit

# Compiler settings
NVCC = nvcc
CC = gcc
CFLAGS = -O3 -lm -std=c99
NVCCFLAGS = -O3 -arch=sm_60 -std=c++11

# Check for CUDA installation
CUDA_PATH := $(shell which nvcc)
ifndef CUDA_PATH
    $(error CUDA not found. Please install NVIDIA CUDA Toolkit)
endif

# Target executables - exactly 5 programs as requested
TARGETS = generate_bin cuda_cubic_encoder channel_sim cuda_cubic_decoder comparator

# Default target
all: check_cuda $(TARGETS)

# Check CUDA installation
check_cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version | head -4
	@echo "GPU information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

# 1. Generate random binary data (CPU)
generate_bin: generate_bin.c
	$(CC) -O3 -std=c99 $< -o $@

# 2. CUDA Cubic encoder (GPU accelerated)
cuda_cubic_encoder: cuda_cubic_encoder.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 3. Channel simulator (CPU) - math library must come after source
channel_sim: channel_sim.c
	$(CC) -O3 -std=c99 $< -o $@ -lm

# 4. CUDA Cubic decoder (GPU accelerated)
cuda_cubic_decoder: cuda_cubic_decoder.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 5. Comparator (CPU)
comparator: comparator.c
	$(CC) -O3 -std=c99 $< -o $@

# Test the complete pipeline
test: all
	@echo "=== Testing CUDA Cubic Code Pipeline ==="
	@echo "1. Generating random data..."
	./generate_bin
	@echo "2. CUDA encoding..."
	./cuda_cubic_encoder original_data.bin encoded_cubic.bin
	@echo "3. Simulating AWGN channel..."
	./channel_sim encoded_cubic.bin corrupted_llrs_cubic.bin 2.0 3
	@echo "4. CUDA decoding..."
	./cuda_cubic_decoder corrupted_llrs_cubic.bin decoded_cubic.bin
	@echo "5. Comparing results..."
	./comparator original_data.bin decoded_cubic.bin
	@echo "=== Test Complete ==="

# Test multiple SNR values
test_snr: all
	@echo "=== Testing Multiple SNR Values ==="
	./generate_bin
	@for snr in 1.0 1.5 2.0 2.5; do \
		echo "Testing SNR = $$snr dB"; \
		./cuda_cubic_encoder original_data.bin encoded_$$snr.bin; \
		./channel_sim encoded_$$snr.bin llr_$$snr.bin $$snr 3; \
		./cuda_cubic_decoder llr_$$snr.bin decoded_$$snr.bin; \
		echo "Results for SNR = $$snr dB:"; \
		./comparator original_data.bin decoded_$$snr.bin; \
		echo ""; \
	done
	@echo "=== SNR Test Complete ==="

# Performance benchmark
benchmark: all
	@echo "=== Performance Benchmark ==="
	@echo "Generating large dataset..."
	# Create larger file for benchmarking (50 blocks = 6250 bytes)
	dd if=/dev/urandom of=large_data.bin bs=6250 count=1 2>/dev/null
	@echo "CUDA Encoding benchmark..."
	time ./cuda_cubic_encoder large_data.bin encoded_large.bin
	@echo "Channel simulation..."
	time ./channel_sim encoded_large.bin llr_large.bin 1.5 3
	@echo "CUDA Decoding benchmark..."
	time ./cuda_cubic_decoder llr_large.bin decoded_large.bin
	@echo "Verification..."
	./comparator large_data.bin decoded_large.bin
	@echo "=== Benchmark Complete ==="

# Memory usage test
test_memory: all
	@echo "=== Memory Usage Test ==="
	./generate_bin
	./cuda_cubic_encoder original_data.bin encoded_mem.bin
	./channel_sim encoded_mem.bin llr_mem.bin 2.0 3
	@echo "Monitoring GPU memory during decoding..."
	@nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > mem_before.txt 2>/dev/null || echo "0" > mem_before.txt
	./cuda_cubic_decoder llr_mem.bin decoded_mem.bin
	@nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > mem_after.txt 2>/dev/null || echo "0" > mem_after.txt
	@echo "GPU Memory: Before=$(shell cat mem_before.txt) MB, After=$(shell cat mem_after.txt) MB"
	./comparator original_data.bin decoded_mem.bin
	@rm -f mem_before.txt mem_after.txt
	@echo "=== Memory Test Complete ==="

# Validate against MATLAB (if available)
validate_matlab: all
	@echo "=== MATLAB Validation (if available) ==="
	@if command -v matlab >/dev/null 2>&1; then \
		echo "MATLAB found, running validation..."; \
		matlab -batch "try; validate_cuda_implementation; catch e; disp(e.message); end"; \
	else \
		echo "MATLAB not found, skipping validation"; \
		echo "To validate, install MATLAB and run: matlab -batch 'validate_cuda_implementation'"; \
	fi

# Race condition and deadlock detection
test_race_conditions: all
	@echo "=== Testing for Race Conditions ==="
	@echo "Running multiple iterations to detect race conditions..."
	@for i in 1 2 3 4 5; do \
		echo "Iteration $$i:"; \
		./generate_bin; \
		./cuda_cubic_encoder original_data.bin encoded_race$$i.bin; \
		./channel_sim encoded_race$$i.bin llr_race$$i.bin 2.0 3; \
		./cuda_cubic_decoder llr_race$$i.bin decoded_race$$i.bin; \
		./comparator original_data.bin decoded_race$$i.bin; \
	done
	@echo "=== Race Condition Test Complete ==="

# Clean build artifacts and temporary files
clean:
	rm -f $(TARGETS)
	rm -f *.bin *.txt *.o
	@echo "Cleaned all build artifacts and temporary files"

# Clean only temporary files, keep executables
clean_temp:
	rm -f *.bin *.txt
	@echo "Cleaned temporary files"

# Install CUDA (Ubuntu/Debian)
install_cuda:
	@echo "To install CUDA on Ubuntu/Debian:"
	@echo "sudo apt update"
	@echo "sudo apt install nvidia-cuda-toolkit nvidia-driver-XXX"
	@echo "Replace XXX with your GPU driver version"
	@echo "Then reboot and run 'make check_cuda'"

# Help target
help:
	@echo "CUDA Cubic Decoder Makefile"
	@echo "============================"
	@echo "Available targets:"
	@echo "  all                 - Build all 5 programs"
	@echo "  test                - Run basic pipeline test"
	@echo "  test_snr            - Test multiple SNR values"
	@echo "  benchmark           - Performance benchmark"
	@echo "  test_memory         - GPU memory usage test"
	@echo "  test_race_conditions - Test for race conditions"
	@echo "  validate_matlab     - Validate against MATLAB (if available)"
	@echo "  clean               - Remove all build artifacts"
	@echo "  clean_temp          - Remove temporary files only"
	@echo "  check_cuda          - Check CUDA installation"
	@echo "  install_cuda        - Show CUDA installation instructions"
	@echo "  help                - Show this help"
	@echo ""
	@echo "Individual programs:"
	@echo "  generate_bin        - Random data generator"
	@echo "  cuda_cubic_encoder  - CUDA-accelerated encoder"
	@echo "  channel_sim         - AWGN channel simulator"
	@echo "  cuda_cubic_decoder  - CUDA-accelerated decoder"
	@echo "  comparator          - File comparison tool"
	@echo ""
	@echo "Usage example:"
	@echo "  make test           # Run complete pipeline test"
	@echo "  make benchmark      # Performance test"

# Debug build (with debug symbols)
debug: NVCCFLAGS += -g -G -lineinfo
debug: CFLAGS += -g -DDEBUG
debug: clean all
	@echo "Debug build complete. Use cuda-gdb for debugging."

# Profile build (for nvprof)
profile: NVCCFLAGS += -lineinfo
profile: clean all
	@echo "Profile build complete. Use 'nvprof ./cuda_cubic_decoder ...' for profiling."

.PHONY: all test test_snr benchmark test_memory test_race_conditions validate_matlab clean clean_temp check_cuda install_cuda help debug profile
