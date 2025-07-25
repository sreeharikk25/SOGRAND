#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Test to diagnose CUDA decoder issues
int main(int argc, char *argv[]) {
    const int n = 15;
    const int k = 10;
    const int block_size = n * n * n;
    
    // Create a simple test pattern
    double* test_llrs = (double*)malloc(block_size * sizeof(double));
    
    // Initialize with strong positive LLRs (should decode to all zeros)
    for (int i = 0; i < block_size; i++) {
        test_llrs[i] = 10.0; // Strong indication of bit = 0
    }
    
    // Write test file
    FILE* f = fopen("test_llrs.bin", "wb");
    fwrite(test_llrs, sizeof(double), block_size, f);
    fclose(f);
    
    printf("Test 1: All positive LLRs (+10.0)\n");
    printf("Expected: All zeros in decoded output\n");
    
    // Create another test with negative LLRs
    for (int i = 0; i < block_size; i++) {
        test_llrs[i] = -10.0; // Strong indication of bit = 1
    }
    
    f = fopen("test_llrs_neg.bin", "wb");
    fwrite(test_llrs, sizeof(double), block_size, f);
    fclose(f);
    
    printf("\nTest 2: All negative LLRs (-10.0)\n");
    printf("Expected: Mostly ones in systematic part\n");
    
    // Test 3: Create a valid codeword pattern
    // First create all zeros (which should be a valid codeword)
    for (int i = 0; i < block_size; i++) {
        test_llrs[i] = 5.0;
    }
    
    f = fopen("test_llrs_valid.bin", "wb");
    fwrite(test_llrs, sizeof(double), block_size, f);
    fclose(f);
    
    printf("\nTest 3: Valid all-zeros codeword\n");
    printf("Expected: All zeros in decoded output\n");
    
    // Test 4: Check tensor indexing
    printf("\nTest 4: Tensor indexing verification\n");
    printf("C-style indexing: idx = k*n*n + j*n + i\n");
    
    // Test systematic bit positions for first few bits
    printf("\nSystematic bit positions (first 10):\n");
    for (int bit = 0; bit < 10; bit++) {
        int slice = bit / (k * k);
        int row = (bit % (k * k)) / k;
        int col = bit % k;
        int tensor_idx = slice * n * n + row * n + col;
        printf("Bit %d: slice=%d, row=%d, col=%d -> tensor_idx=%d\n", 
               bit, slice, row, col, tensor_idx);
    }
    
    free(test_llrs);
    
    printf("\nRun the decoders on these test files to diagnose the issue:\n");
    printf("./cubic_decoder_fixed test_llrs.bin test_out1.bin\n");
    printf("./cubic_decoder_fixed test_llrs_neg.bin test_out2.bin\n");
    printf("./cubic_decoder_fixed test_llrs_valid.bin test_out3.bin\n");
    
    return 0;
}