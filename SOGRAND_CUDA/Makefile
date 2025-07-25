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
CUDA_TARGETS = channel_sim square_encoder square_decoder \
               cubic_encoder cubic_decoder \
               square_decoder_fixed cubic_decoder_fixed \
               cubic_encoder_fixed

C_TARGETS = generate_bin comparator

ALL_TARGETS = $(CUDA_TARGETS) $(C_TARGETS)

# Default target
all: $(ALL_TARGETS)

# CUDA programs
channel_sim: channel_sim.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

square_encoder: square_encoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

square_decoder: square_decoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

square_decoder_fixed: square_decoder_fixed.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cubic_encoder: cubic_encoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cubic_encoder_fixed: cubic_encoder_fixed.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cubic_decoder: cubic_decoder.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

cubic_decoder_fixed: cubic_decoder_fixed.cu
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
	./square_encoder original_data.bin encoded_square.bin
	./channel_sim encoded_square.bin corrupted_llrs_square.bin 2.0 2
	./square_decoder corrupted_llrs_square.bin decoded_square.bin
	./comparator original_data.bin decoded_square.bin

test_square_fixed: all
	@echo "Running fixed square code test..."
	./generate_bin
	./square_encoder original_data.bin encoded_square.bin
	./channel_sim encoded_square.bin corrupted_llrs_square.bin 2.0 2
	./square_decoder_fixed corrupted_llrs_square.bin decoded_square_fixed.bin
	./comparator original_data.bin decoded_square_fixed.bin

test_cubic: all
	@echo "Running cubic code test..."
	./generate_bin
	./cubic_encoder original_data.bin encoded_cubic.bin
	./channel_sim encoded_cubic.bin corrupted_llrs_cubic.bin 2.0 3
	./cubic_decoder corrupted_llrs_cubic.bin decoded_cubic.bin
	./comparator original_data.bin decoded_cubic.bin

test_cubic_fixed: all
	@echo "Running fixed cubic code test..."
	./generate_bin
	./cubic_encoder_fixed original_data.bin encoded_cubic_fixed.bin
	./channel_sim encoded_cubic_fixed.bin corrupted_llrs_cubic_fixed.bin 2.0 3
	./cubic_decoder_fixed corrupted_llrs_cubic_fixed.bin decoded_cubic_fixed.bin
	./comparator original_data.bin decoded_cubic_fixed.bin

# Performance comparison
benchmark: all
	@echo "Benchmarking CUDA vs CPU implementations..."
	@echo "This would require the original CPU versions to be compiled as well"

.PHONY: all clean install test_square test_cubic test_square_fixed test_cubic_fixed benchmark
