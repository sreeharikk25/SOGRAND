#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Generator matrix in constant memory
__constant__ int d_G[25*31];

// Kernel for encoding rows
__global__ void encode_rows_kernel(int* input, int* output, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || row >= k) return;
    
    int in_offset = block_id * k * k;
    int out_offset = block_id * n * n;
    
    // Encode this row
    for (int col = 0; col < n; col++) {
        int sum = 0;
        for (int j = 0; j < k; j++) {
            sum += input[in_offset + row * k + j] * d_G[j * n + col];
        }
        output[out_offset + row * n + col] = sum % 2;
    }
}

// Kernel for encoding columns
__global__ void encode_columns_kernel(int* codeword, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || col >= n) return;
    
    int offset = block_id * n * n;
    
    // Extract column data from first k rows
    int temp_col[25];  // Max k = 25
    for (int i = 0; i < k; i++) {
        temp_col[i] = codeword[offset + i * n + col];
    }
    
    // Encode and fill remaining rows
    for (int row = k; row < n; row++) {
        int sum = 0;
        for (int j = 0; j < k; j++) {
            sum += temp_col[j] * d_G[j * n + row];
        }
        codeword[offset + row * n + col] = sum % 2;
    }
}

// Host function to encode multiple blocks
void encode_square_cuda(unsigned char* h_input, unsigned char* h_output, 
                       int num_blocks, int n, int k) {
    int message_block_size = k * k;
    int codeword_block_size = n * n;
    
    // Convert bytes to bits for processing
    int* h_input_bits = (int*)malloc(num_blocks * message_block_size * sizeof(int));
    int* h_output_bits = (int*)malloc(num_blocks * codeword_block_size * sizeof(int));
    
    // Convert input bytes to bits
    for (int b = 0; b < num_blocks; b++) {
        for (int byte_idx = 0; byte_idx < (message_block_size + 7) / 8; byte_idx++) {
            unsigned char byte = h_input[b * ((message_block_size + 7) / 8) + byte_idx];
            for (int bit_idx = 0; bit_idx < 8 && (byte_idx * 8 + bit_idx) < message_block_size; bit_idx++) {
                h_input_bits[b * message_block_size + byte_idx * 8 + bit_idx] = 
                    (byte >> (7 - bit_idx)) & 1;
            }
        }
    }
    
    // Allocate device memory
    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, num_blocks * message_block_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, num_blocks * codeword_block_size * sizeof(int)));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input_bits, 
                          num_blocks * message_block_size * sizeof(int),
                          cudaMemcpyHostToDevice));
    
    // Setup grid dimensions
    dim3 threadsPerBlock(32);
    dim3 blocksPerGridRows((k + threadsPerBlock.x - 1) / threadsPerBlock.x, num_blocks);
    dim3 blocksPerGridCols((n + threadsPerBlock.x - 1) / threadsPerBlock.x, num_blocks);
    
    // Encode rows
    encode_rows_kernel<<<blocksPerGridRows, threadsPerBlock>>>(
        d_input, d_output, n, k, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Encode columns
    encode_columns_kernel<<<blocksPerGridCols, threadsPerBlock>>>(
        d_output, n, k, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output_bits, d_output,
                          num_blocks * codeword_block_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
    
    // Convert bits back to bytes
    for (int b = 0; b < num_blocks; b++) {
        int bit_count = 0;
        unsigned char byte = 0;
        for (int i = 0; i < codeword_block_size; i++) {
            byte = (byte << 1) | h_output_bits[b * codeword_block_size + i];
            bit_count++;
            if (bit_count == 8) {
                h_output[b * ((codeword_block_size + 7) / 8) + (i / 8)] = byte;
                byte = 0;
                bit_count = 0;
            }
        }
        if (bit_count > 0) {
            byte <<= (8 - bit_count);
            h_output[b * ((codeword_block_size + 7) / 8) + (codeword_block_size / 8)] = byte;
        }
    }
    
    // Cleanup
    free(h_input_bits);
    free(h_output_bits);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// CRC polynomial functions (same as original)
int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) {
        poly[i] = (dec_val >> (len - 1 - i)) & 1;
    }
    poly[len] = 1;
    return poly;
}

void getGH_sys_CRC(int n, int k, int* G_flat) {
    const char* hex_poly = NULL;
    int r = n - k;
    
    if (r == 3) hex_poly = "0x5";
    else if (r == 4) hex_poly = "0x9";
    else if (r == 5 && k <= 10) hex_poly = "0x15";
    else if (r == 5 && k <= 26) hex_poly = "0x12";
    else if (r == 6 && k <= 25) hex_poly = "0x23";
    else if (r == 6 && k <= 57) hex_poly = "0x33";
    else {
        fprintf(stderr, "Error: (n, k) = (%d, %d) is not supported.\n", n, k);
        exit(1);
    }
    
    int poly_len;
    int* poly = koopman2matlab(hex_poly, &poly_len);
    
    // Generate parity matrix P
    int** P = (int**)malloc(k * sizeof(int*));
    for(int i = 0; i < k; i++) P[i] = (int*)malloc(r * sizeof(int));
    int* msg_poly = (int*)calloc(k + r, sizeof(int));
    
    for (int i = 0; i < k; i++) {
        memset(msg_poly, 0, (k + r) * sizeof(int));
        msg_poly[i] = 1;
        
        for (int j = 0; j < k; j++) {
            if (msg_poly[j] == 1) {
                for (int l = 0; l < poly_len; l++) {
                    msg_poly[j + l] ^= poly[l];
                }
            }
        }
        for (int j = 0; j < r; j++) P[i][j] = msg_poly[k + j];
    }
    
    // Build generator matrix G = [I_k | P]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            G_flat[i * n + j] = (i == j) ? 1 : 0;
        }
        for (int j = 0; j < r; j++) {
            G_flat[i * n + k + j] = P[i][j];
        }
    }
    
    // Cleanup
    free(poly);
    free(msg_poly);
    for(int i = 0; i < k; i++) free(P[i]);
    free(P);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    const int n = 31;
    const int k = 25;
    const int message_block_size = k * k;     // 625 bits
    const int codeword_block_size = n * n;    // 961 bits
    const int message_bytes = (message_block_size + 7) / 8;  // 79 bytes
    const int codeword_bytes = (codeword_block_size + 7) / 8; // 121 bytes

    // Initialize generator matrix
    int* h_G = (int*)malloc(k * n * sizeof(int));
    getGH_sys_CRC(n, k, h_G);
    
    // Copy to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_G, h_G, k * n * sizeof(int)));
    free(h_G);

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Square Encoder (n=%d, k=%d)...\n", n, k);

    // Process in batches for better GPU utilization
    const int BATCH_SIZE = 64;
    unsigned char* input_batch = (unsigned char*)malloc(BATCH_SIZE * message_bytes);
    unsigned char* output_batch = (unsigned char*)malloc(BATCH_SIZE * codeword_bytes);
    
    long total_blocks_encoded = 0;
    
    while (!feof(fin)) {
        // Read batch of message blocks
        int blocks_read = 0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            size_t bytes_read = fread(&input_batch[i * message_bytes], 1, message_bytes, fin);
            if (bytes_read == 0) break;
            
            // Pad with zeros if partial block
            if (bytes_read < message_bytes) {
                memset(&input_batch[i * message_bytes + bytes_read], 0, 
                       message_bytes - bytes_read);
            }
            blocks_read++;
        }
        
        if (blocks_read == 0) break;
        
        // Encode batch on GPU
        encode_square_cuda(input_batch, output_batch, blocks_read, n, k);
        
        // Write encoded blocks
        fwrite(output_batch, codeword_bytes, blocks_read, fout);
        total_blocks_encoded += blocks_read;
    }

    printf("Encoding complete. %ld block(s) encoded.\n", total_blocks_encoded);

    free(input_batch);
    free(output_batch);
    fclose(fin);
    fclose(fout);

    return 0;
}
