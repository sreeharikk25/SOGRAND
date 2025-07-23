#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// CUDA kernel for AWGN channel simulation
__global__ void channel_simulation_kernel(unsigned char* input_bits, double* output_llrs, 
                                         int num_bits, double sigma, double variance,
                                         curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bits) return;
    
    // Initialize random state for this thread
    curand_init(clock64(), idx, 0, &states[idx]);
    
    // Get bit value
    int byte_idx = idx / 8;
    int bit_idx = 7 - (idx % 8);
    int bit = (input_bits[byte_idx] >> bit_idx) & 1;
    
    // BPSK modulation
    double x = (bit == 0) ? 1.0 : -1.0;
    
    // Generate Gaussian noise
    double noise = curand_normal_double(&states[idx]) * sigma;
    
    // Add noise
    double y = x + noise;
    
    // Calculate LLR
    output_llrs[idx] = 2.0 * y / variance;
}

// Host function to setup and launch kernel
void simulate_channel_cuda(unsigned char* h_input, double* h_output, 
                          int num_bytes, double sigma, double variance) {
    int num_bits = num_bytes * 8;
    
    // Allocate device memory
    unsigned char* d_input;
    double* d_output;
    curandState* d_states;
    
    CHECK_CUDA(cudaMalloc(&d_input, num_bytes * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_output, num_bits * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_states, num_bits * sizeof(curandState)));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, num_bytes * sizeof(unsigned char), 
                          cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blocks = (num_bits + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    channel_simulation_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_input, d_output, num_bits, sigma, variance, d_states);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, num_bits * sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_states));
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <Eb/N0_in_dB> <code_dimension>\n", argv[0]);
        fprintf(stderr, "  <code_dimension> should be 2 for square code, 3 for cubic code.\n");
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    double EbN0dB = atof(argv[3]);
    int code_dim = atoi(argv[4]);

    if (code_dim != 2 && code_dim != 3) {
        fprintf(stderr, "Error: Code dimension must be 2 or 3.\n");
        return 1;
    }

    // Code parameters
    int n, k;
    if (code_dim == 2) {
        n = 31; k = 25; // Square code
    } else {
        n = 15; k = 10; // Cubic code
    }
    
    const double R = pow((double)k / n, code_dim);
    double EsN0dB = EbN0dB + 10 * log10(2 * R);
    double sigma = 1.0 / sqrt(pow(10, EsN0dB / 10.0));
    double variance = sigma * sigma;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    printf("CUDA Channel Simulation: Eb/N0 = %.2f dB (sigma = %f)...\n", EbN0dB, sigma);

    // Process file in chunks for better GPU utilization
    const int CHUNK_SIZE = 1024 * 1024; // 1MB chunks
    unsigned char* input_buffer = (unsigned char*)malloc(CHUNK_SIZE);
    double* output_buffer = (double*)malloc(CHUNK_SIZE * 8 * sizeof(double));
    
    size_t bytes_read;
    while ((bytes_read = fread(input_buffer, 1, CHUNK_SIZE, fin)) > 0) {
        // Run CUDA simulation
        simulate_channel_cuda(input_buffer, output_buffer, bytes_read, sigma, variance);
        
        // Write output
        fwrite(output_buffer, sizeof(double), bytes_read * 8, fout);
    }

    printf("Channel simulation complete. Output file '%s' contains LLRs.\n", output_filename);

    free(input_buffer);
    free(output_buffer);
    fclose(fin);
    fclose(fout);

    return 0;
}
