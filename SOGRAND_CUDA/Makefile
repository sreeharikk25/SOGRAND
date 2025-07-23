# CUDA Makefile for Product Code Implementation

# CUDA compiler
NVCC = nvcc

# CUDA flags
CUDA_FLAGS = -O3 -arch=sm_70 -std=c++14
CUDA_LIBS = -lcudart -lcurand

# C compiler for non-CUDA programs
CC = gcc
CFLAGS = -O3 -lm

# Target executables
CUDA_TARGETS = cuda_channel_sim cuda_square_encoder cuda_square_decoder \
               cuda_cubic_encoder cuda_cubic_decoder

C_TARGETS = generate_bin comparator

ALL_TARGETS = $(CUDA_TARGETS) $(C_TARGETS)

# Default target
all: $(ALL_TARGETS)

# CUDA programs
cuda_channel_sim: cuda_channel_sim.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cuda_square_encoder: cuda_square_encoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cuda_square_decoder: cuda_square_decoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cuda_cubic_encoder: cuda_cubic_encoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cuda_cubic_decoder: cuda_cubic_decoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

# C programs (unchanged)
generate_bin: generate_bin.c
	$(CC) $(CFLAGS) -o $@ $<

comparator: comparator.c
	$(CC) $(CFLAGS) -o $@ $<

# Clean
clean:
	rm -f $(ALL_TARGETS) *.o

# Install (optional)
install: all
	@echo "Installing CUDA product code binaries..."
	@mkdir -p ~/bin
	@cp $(ALL_TARGETS) ~/bin/
	@echo "Installation complete. Add ~/bin to your PATH if needed."

# Test targets
test_square: all
	@echo "Running square code test..."
	./generate_bin
	./cuda_square_encoder original_data.bin encoded_square.bin
	./cuda_channel_sim encoded_square.bin corrupted_llrs_square.bin 2.0 2
	./cuda_square_decoder corrupted_llrs_square.bin decoded_square.bin
	./comparator original_data.bin decoded_square.bin

test_cubic: all
	@echo "Running cubic code test..."
	./generate_bin
	./cuda_cubic_encoder original_data.bin encoded_cubic.bin
	./cuda_channel_sim encoded_cubic.bin corrupted_llrs_cubic.bin 2.0 3
	./cuda_cubic_decoder corrupted_llrs_cubic.bin decoded_cubic.bin
	./comparator original_data.bin decoded_cubic.bin

# Performance comparison
benchmark: all
	@echo "Benchmarking CUDA vs CPU implementations..."
	@echo "This would require the original CPU versions to be compiled as well"

.PHONY: all clean install test_square test_cubic benchmark
