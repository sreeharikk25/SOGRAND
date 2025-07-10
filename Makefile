# Define the C compiler to use
CC = gcc

# Define common compilation flags
# -Wall: Enable all standard warnings
# -Wextra: Enable extra warnings
# -g: Include debugging information
# -O2: Optimize code for speed
CFLAGS = -Wall -Wextra -g -O2

# Define the source files
SRCS = \
	square_encoder.c \
	cubic_encoder.c \
	cubic_decoder.c \
	square_decoder.c \
	generate_bin.c \
	channel_sim.c \
	comparator.c

# Define the executable targets
BINS = \
	square_encoder \
	cubic_encoder \
	cubic_decoder \
	square_decoder \
	generate_bin \
	channel_sim \
	comparator

# Define the object files (automatically generated from SRCS)
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

# Default target: build all executables
all: $(BINS)

# Rule to compile a .c file into an executable
# $<: the first prerequisite (the .c file)
# $@: the target (the executable name)
%: %.c
	$(CC) $(CFLAGS) $< -o $@

# Rule to clean up generated files
clean:
	rm -f $(BINS) $(OBJS)
	rm -f *.bin # Assuming generate_bin might create .bin files
