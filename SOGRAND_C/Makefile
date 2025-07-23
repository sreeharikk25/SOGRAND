# Makefile for Square and Cubic Encoder/Decoder

CC = gcc
CFLAGS = -Wall -Wextra -g -O2
LDFLAGS = -lm
# MPI setup (adjust if needed)
MPICC = mpicc

# Targets
TARGETS = square_encoder cubic_encoder cubic_decoder square_decoder generate_bin channel_sim comparator

all: $(TARGETS)

# Object files
OBJS = square_encoder.o cubic_encoder.o cubic_decoder.o square_decoder.o generate_bin.o channel_sim.o comparator.o

# Rules for each executable

square_encoder: square_encoder.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cubic_encoder: cubic_encoder.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cubic_decoder: cubic_decoder.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

square_decoder: square_decoder.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

generate_bin: generate_bin.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

channel_sim: channel_sim.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

comparator: comparator.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGETS) $(OBJS)
	rm -f *.bin # Assuming generate_bin might create .bin files

.PHONY: all clean
