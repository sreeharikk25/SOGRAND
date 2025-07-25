#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 128
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Generator matrix in constant memory
__constant__ int d_G[10*15];

// 3D tensor access - both host and device versions
__host__ __device__ inline int tensor_idx(int i, int j, int k, int n) {
    return k * n * n + j * n + i;
}

// Kernel for encoding rows (stage 1)
__global__ void encode_rows_cubic_kernel(int* codeword, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || slice >= k || row >= k) return;
    
    int offset = block_id * n * n * n;
    
    // Extract row data
    int temp_vec[10];  // Max k = 10
    for (int j = 0; j < k; j++) {
        temp_vec[j] = codeword[offset + tensor_idx(row, j, slice, n)];
    }
    
    // Encode parity columns
    for (int col = k; col < n; col++) {
        int sum = 0;
        for (int j = 0; j < k; j++) {
            sum += temp_vec[j] * d_G[j * n + col];
        }
        codeword[offset + tensor_idx(row, col, slice, n)] = sum % 2;
    }
}

// Kernel for encoding columns (stage 2)
__global__ void encode_columns_cubic_kernel(int* codeword, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || slice >= k || col >= n) return;
    
    int offset = block_id * n * n * n;
    
    // Extract column data
    int temp_vec[10];
    for (int i = 0; i < k; i++) {
        temp_vec[i] = codeword[offset + tensor_idx(i, col, slice, n)];
    }
    
    // Encode parity rows
    for (int row = k; row < n; row++) {
        int sum = 0;
        for (int j = 0; j < k; j++) {
            sum += temp_vec[j] * d_G[j * n + row];
        }
        codeword[offset + tensor_idx(row, col, slice, n)] = sum % 2;
    }
}

// Kernel for encoding slices (stage 3)
__global__ void encode_slices_cubic_kernel(int* codeword, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || row >= n || col >= n) return;
    
    int offset = block_id * n * n * n;
    
    // Extract slice vector
    int temp_vec[10];
    for (int i = 0; i < k; i++) {
        temp_vec[i] = codeword[offset + tensor_idx(row, col, i, n)];
    }
    
    // Encode parity slices
    for (int slice = k; slice < n; slice++) {
        int sum = 0;
        for (int j = 0; j < k; j++) {
            sum += temp_vec[j] * d_G[j * n + slice];
        }
        codeword[offset + tensor_idx(row, col, slice, n)] = sum % 2;
    }
}

// FIXED: Kernel to copy systematic part with correct bit ordering
__global__ void copy_systematic_kernel_fixed(int* input, int* output, int n, int k, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * k * k * k;
    
    if (idx >= total_elements) return;
    
    int block_id = idx / (k * k * k);
    int local_idx = idx % (k * k * k);
    
    // FIXED: Correct mapping to match C code tensor layout
    // In C: tensor.data[k*dim1*dim2 + j*dim1 + i] for (i,j,k) position
    // So for k×k×k tensor: index = slice*k*k + col*k + row
    int row = local_idx % k;
    int col = (local_idx / k) % k;
    int slice = local_idx / (k * k);
    
    int in_offset = block_id * k * k * k;
    int out_offset = block_id * n * n * n;
    
    output[out_offset + tensor_idx(row, col, slice, n)] = input[in_offset + local_idx];
}

// Host-side CRC matrix generation
int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) poly[i] = (dec_val >> (len - 1 - i)) & 1;
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

    // Temporary matrices
    int** P = (int**)malloc(k * sizeof(int*));
    for(int i=0; i<k; ++i) P[i] = (int*)calloc(r, sizeof(int));
    int* msg_poly = (int*)calloc(k + r, sizeof(int));

    // Generate parity matrix P
    for (int i = 0; i < k; i++) {
        memset(msg_poly, 0, (k + r) * sizeof(int));
        msg_poly[i] = 1;

        // Polynomial division
        for (int j = 0; j < k; j++) {
            if (msg_poly[j] == 1) {
                for (int l = 0; l < poly_len; l++) {
                    msg_poly[j + l] ^= poly[l];
                }
            }
        }
        
        // Extract remainder (parity bits)
        for (int j = 0; j < r; j++) P[i][j] = msg_poly[k + j];
    }

    // Build G matrix: G = [I_k | P]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            G_flat[i * n + j] = (i == j) ? 1 : 0;
        }
        for (int j = 0; j < r; j++) {
            G_flat[i * n + (k + j)] = P[i][j];
        }
    }

    // Cleanup
    free(poly);
    free(msg_poly);
    for(int i=0; i<k; ++i) free(P[i]);
    free(P);
}

// Host function to encode multiple blocks
void encode_cubic_cuda_fixed(unsigned char* h_input, unsigned char* h_output, 
                            int num_blocks, int n, int k) {
    int message_block_size = k * k * k;
    int codeword_block_size = n * n * n;
    
    // Convert bytes to bits
    int* h_input_bits = (int*)malloc(num_blocks * message_block_size * sizeof(int));
    int* h_output_bits = (int*)malloc(num_blocks * codeword_block_size * sizeof(int));
    
    // Convert input bytes to bits
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < message_block_size; i++) {
            int byte_idx = i / 8;
            int bit_idx = 7 - (i % 8);
            h_input_bits[b * message_block_size + i] = 
                (h_input[b * ((message_block_size + 7) / 8) + byte_idx] >> bit_idx) & 1;
        }
    }
    
    // Allocate device memory
    int *d_input, *d_codeword;
    CHECK_CUDA(cudaMalloc(&d_input, num_blocks * message_block_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_codeword, num_blocks * codeword_block_size * sizeof(int)));
    
    // Initialize codeword to zero
    CHECK_CUDA(cudaMemset(d_codeword, 0, num_blocks * codeword_block_size * sizeof(int)));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input_bits, 
                          num_blocks * message_block_size * sizeof(int),
                          cudaMemcpyHostToDevice));
    
    // Copy systematic part with fixed kernel
    int threads = 256;
    int blocks = (num_blocks * message_block_size + threads - 1) / threads;
    copy_systematic_kernel_fixed<<<blocks, threads>>>(d_input, d_codeword, n, k, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Setup grid dimensions for encoding
    dim3 threadsPerBlock(16);
    dim3 blocksRows((k + threadsPerBlock.x - 1) / threadsPerBlock.x, k, num_blocks);
    dim3 blocksCols((n + threadsPerBlock.x - 1) / threadsPerBlock.x, k, num_blocks);
    dim3 blocksSlices((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);
    
    // Stage 1: Encode rows
    encode_rows_cubic_kernel<<<blocksRows, threadsPerBlock>>>(d_codeword, n, k, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Stage 2: Encode columns
    encode_columns_cubic_kernel<<<blocksCols, threadsPerBlock>>>(d_codeword, n, k, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Stage 3: Encode slices
    encode_slices_cubic_kernel<<<blocksSlices, threadsPerBlock>>>(d_codeword, n, k, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output_bits, d_codeword,
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
    
    // Free memory
    free(h_input_bits);
    free(h_output_bits);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_codeword));
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    const int n = 15;
    const int k = 10;
    const int message_block_size = k * k * k;
    const int codeword_block_size = n * n * n;

    // Generate CRC matrix
    int h_G[10*15];
    getGH_sys_CRC(n, k, h_G);
    
    // Copy to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_G, h_G, sizeof(h_G)));

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Cubic Encoder Fixed (n=%d, k=%d)...\n", n, k);

    // Process in batches
    const int BATCH_SIZE = 64;
    int message_bytes_per_block = (message_block_size + 7) / 8;
    int codeword_bytes_per_block = (codeword_block_size + 7) / 8;
    
    unsigned char* input_buffer = (unsigned char*)malloc(BATCH_SIZE * message_bytes_per_block);
    unsigned char* output_buffer = (unsigned char*)malloc(BATCH_SIZE * codeword_bytes_per_block);
    
    int total_blocks = 0;
    
    while (true) {
        int blocks_read = 0;
        
        // Read blocks
        for (int i = 0; i < BATCH_SIZE; i++) {
            size_t bytes_read = fread(&input_buffer[i * message_bytes_per_block], 
                                     1, message_bytes_per_block, fin);
            if (bytes_read == 0) break;
            
            // Pad if necessary
            if (bytes_read < message_bytes_per_block) {
                memset(&input_buffer[i * message_bytes_per_block + bytes_read], 
                       0, message_bytes_per_block - bytes_read);
            }
            blocks_read++;
        }
        
        if (blocks_read == 0) break;
        
        // Encode batch
        encode_cubic_cuda_fixed(input_buffer, output_buffer, blocks_read, n, k);
        
        // Write output
        fwrite(output_buffer, codeword_bytes_per_block, blocks_read, fout);
        
        total_blocks += blocks_read;
    }

    printf("Encoding complete. %d block(s) encoded.\n", total_blocks);

    free(input_buffer);
    free(output_buffer);
    fclose(fin);
    fclose(fout);

    return 0;
}