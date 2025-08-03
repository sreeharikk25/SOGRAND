#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// --- CUDA Kernels for Encoding ---
__global__ void encode_rows_kernel(int* c_data, int* G_flat, int n, int k) {
    int slice = blockIdx.x;
    int row = threadIdx.x;

    if (slice >= k || row >= k) return;

    // Encode row: calculate parity for columns k through n-1
    for (int col = k; col < n; col++) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
            int msg_bit_idx_3d = slice * n * n + row * n + msg_bit_idx;
            parity_val += c_data[msg_bit_idx_3d] * G_flat[msg_bit_idx * n + col];
        }
        int col_idx_3d = slice * n * n + row * n + col;
        c_data[col_idx_3d] = parity_val % 2;
    }
}

__global__ void encode_columns_kernel(int* c_data, int* G_flat, int n, int k) {
    int slice = blockIdx.x;
    int col = threadIdx.x;

    if (slice >= k || col >= n) return;

    // Encode column: calculate parity for rows k through n-1
    for (int row = k; row < n; row++) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
            int msg_bit_idx_3d = slice * n * n + msg_bit_idx * n + col;
            parity_val += c_data[msg_bit_idx_3d] * G_flat[msg_bit_idx * n + row];
        }
        int row_idx_3d = slice * n * n + row * n + col;
        c_data[row_idx_3d] = parity_val % 2;
    }
}

__global__ void encode_slices_kernel(int* c_data, int* G_flat, int n, int k) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= n || col >= n) return;

    // Encode slice: calculate parity for slices k through n-1
    for (int slice = k; slice < n; slice++) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
            int msg_bit_idx_3d = msg_bit_idx * n * n + row * n + col;
            parity_val += c_data[msg_bit_idx_3d] * G_flat[msg_bit_idx * n + slice];
        }
        int slice_idx_3d = slice * n * n + row * n + col;
        c_data[slice_idx_3d] = parity_val % 2;
    }
}

// --- Host Functions ---
int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) poly[i] = (dec_val >> (len - 1 - i)) & 1;
    poly[len] = 1;
    return poly;
}

void getGH_sys_CRC(int n, int k, int** G) {
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

    int** P = (int**)malloc(k * sizeof(int*));
    for(int i=0; i<k; ++i) P[i] = (int*)malloc(r * sizeof(int));
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

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) G[i][j] = (i == j) ? 1 : 0;
        for (int j = 0; j < r; j++) G[i][k + j] = P[i][j];
    }

    free(poly);
    free(msg_poly);
    for(int i=0; i<k; ++i) free(P[i]);
    free(P);
}

void encode_block_cuda(int* current_bits, FILE* fout, unsigned char* byte_out, int* bit_count_out, int** G, int n, int k) {
    const int message_block_size = k * k * k;
    const int codeword_block_size = n * n * n;

    // Allocate unified memory for the codeword tensor
    int* d_c;
    CUDA_CHECK(cudaMallocManaged(&d_c, codeword_block_size * sizeof(int)));

    // Flatten and copy G matrix to GPU
    int* d_G_flat;
    CUDA_CHECK(cudaMallocManaged(&d_G_flat, k * n * sizeof(int)));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            d_G_flat[i * n + j] = G[i][j];
        }
    }

    // Initialize codeword with zeros
    CUDA_CHECK(cudaMemset(d_c, 0, codeword_block_size * sizeof(int)));

    // 1. Systematically embed the message u into the codeword c
    int bit_idx = 0;
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                int tensor_idx = slice * n * n + row * n + col;
                d_c[tensor_idx] = current_bits[bit_idx++];
            }
        }
    }

    // Synchronize to ensure systematic part is copied
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Encode rows (write parity to columns k through n-1)
    dim3 grid_rows(k, 1, 1);
    dim3 block_rows(k, 1, 1);
    encode_rows_kernel<<<grid_rows, block_rows>>>(d_c, d_G_flat, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Encode columns (write parity to rows k through n-1)
    dim3 grid_cols(k, 1, 1);
    dim3 block_cols(n, 1, 1);
    encode_columns_kernel<<<grid_cols, block_cols>>>(d_c, d_G_flat, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Encode slices (write parity to slices k through n-1)
    dim3 grid_slices(n, 1, 1);
    dim3 block_slices(n, 1, 1);
    encode_slices_kernel<<<grid_slices, block_slices>>>(d_c, d_G_flat, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Write the full codeword to file
    for (int bit_idx = 0; bit_idx < codeword_block_size; bit_idx++) {
        *byte_out = (*byte_out << 1) | d_c[bit_idx];
        (*bit_count_out)++;
        if (*bit_count_out == 8) {
            fwrite(byte_out, 1, 1, fout);
            *byte_out = 0;
            *bit_count_out = 0;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_G_flat));
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

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));

    int** G = (int**)malloc(k * sizeof(int*));
    for(int i=0; i<k; ++i) G[i] = (int*)malloc(n * sizeof(int));
    getGH_sys_CRC(n, k, G);

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

    printf("CUDA Encoding %s to %s using SYSTEMATIC (n=%d, k=%d) cubic code...\n", input_filename, output_filename, n, k);

    int bit_buffer[message_block_size];
    int bits_in_buffer = 0;
    unsigned char byte_in;
    unsigned char byte_out = 0;
    int bit_count_out = 0;
    long total_blocks_encoded = 0;

    while (fread(&byte_in, 1, 1, fin) == 1) {
        for (int i = 7; i >= 0; i--) {
            bit_buffer[bits_in_buffer++] = (byte_in >> i) & 1;
            if (bits_in_buffer == message_block_size) {
                encode_block_cuda(bit_buffer, fout, &byte_out, &bit_count_out, G, n, k);
                bits_in_buffer = 0;
                total_blocks_encoded++;
            }
        }
    }

    if (bits_in_buffer > 0) {
        for (int i = bits_in_buffer; i < message_block_size; i++) {
            bit_buffer[i] = 0;
        }
        encode_block_cuda(bit_buffer, fout, &byte_out, &bit_count_out, G, n, k);
        total_blocks_encoded++;
    }

    if (bit_count_out > 0) {
        byte_out <<= (8 - bit_count_out);
        fwrite(&byte_out, 1, 1, fout);
    }

    printf("CUDA Encoding complete. %ld block(s) encoded.\n", total_blocks_encoded);

    fclose(fin);
    fclose(fout);
    for(int i=0; i<k; ++i) free(G[i]);
    free(G);

    return 0;
}
