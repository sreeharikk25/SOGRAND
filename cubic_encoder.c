#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Data Structures for 3D Tensor ---
typedef struct {
    int* data;
    int dim1, dim2, dim3;
} Tensor_i;

// --- Forward Declarations ---
Tensor_i create_tensor_i(int d1, int d2, int d3);
void free_tensor_i(Tensor_i t);
int get_tensor_i(Tensor_i t, int i, int j, int k);
void set_tensor_i(Tensor_i t, int i, int j, int k, int val);
int* koopman2matlab(const char* k_poly, int* poly_len);
void getGH_sys_CRC(int n, int k, int** G);
void encode_block(int* current_bits, FILE* fout, unsigned char* byte_out, int* bit_count_out, int** G, int n, int k);


// --- Encoding Function ---
void encode_block(int* current_bits, FILE* fout, unsigned char* byte_out, int* bit_count_out, int** G, int n, int k) {
    const int message_block_size = k * k * k;
    const int codeword_block_size = n * n * n;

    Tensor_i u = create_tensor_i(k, k, k);
    for(int bit_idx=0; bit_idx < message_block_size; ++bit_idx) {
        u.data[bit_idx] = current_bits[bit_idx];
    }

    Tensor_i c = create_tensor_i(n, n, n);
    int temp_vec_k[k];

    // 1. Systematically embed the message u into the codeword c
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                set_tensor_i(c, i, j, l, get_tensor_i(u, i, j, l));
            }
        }
    }

    // 2. Encode rows (write parity to columns k through n-1)
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for(int j=0; j<k; ++j) temp_vec_k[j] = get_tensor_i(c, row, j, slice);
            for (int col = k; col < n; col++) {
                 int parity_val = 0;
                 for (int msg_bit_idx=0; msg_bit_idx<k; ++msg_bit_idx) {
                     parity_val += temp_vec_k[msg_bit_idx] * G[msg_bit_idx][col];
                 }
                 set_tensor_i(c, row, col, slice, parity_val % 2);
            }
        }
    }

    // 3. Encode columns (write parity to rows k through n-1)
    //    This operates on the first k slices, which now contain message and row parity.
    for (int slice = 0; slice < k; slice++) {
        for (int col = 0; col < n; col++) {
            for(int j=0; j<k; ++j) temp_vec_k[j] = get_tensor_i(c, j, col, slice);
            for (int row = k; row < n; row++) {
                int parity_val = 0;
                for (int msg_bit_idx=0; msg_bit_idx<k; ++msg_bit_idx) {
                    parity_val += temp_vec_k[msg_bit_idx] * G[msg_bit_idx][row];
                }
                set_tensor_i(c, row, col, slice, parity_val % 2);
            }
        }
    }

    // 4. Encode slices (write parity to slices k through n-1)
    //    This operates on all n x n planes, which are now all filled.
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
             for(int j=0; j<k; ++j) temp_vec_k[j] = get_tensor_i(c, row, col, j);
             for (int slice = k; slice < n; slice++) {
                 int parity_val = 0;
                 for (int msg_bit_idx=0; msg_bit_idx<k; ++msg_bit_idx) {
                    parity_val += temp_vec_k[msg_bit_idx] * G[msg_bit_idx][slice];
                 }
                 set_tensor_i(c, row, col, slice, parity_val % 2);
            }
        }
    }

    // Write the full codeword to file
    for (int bit_idx = 0; bit_idx < codeword_block_size; bit_idx++) {
        *byte_out = (*byte_out << 1) | c.data[bit_idx];
        (*bit_count_out)++;
        if (*bit_count_out == 8) {
            fwrite(byte_out, 1, 1, fout);
            *byte_out = 0;
            *bit_count_out = 0;
        }
    }
    free_tensor_i(u);
    free_tensor_i(c);
}


// --- Main Application ---
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

    printf("Encoding %s to %s using SYSTEMATIC (n=%d, k=%d) cubic code...\n", input_filename, output_filename, n, k);

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
                encode_block(bit_buffer, fout, &byte_out, &bit_count_out, G, n, k);
                bits_in_buffer = 0;
                total_blocks_encoded++;
            }
        }
    }

    if (bits_in_buffer > 0) {
        for (int i = bits_in_buffer; i < message_block_size; i++) {
            bit_buffer[i] = 0;
        }
        encode_block(bit_buffer, fout, &byte_out, &bit_count_out, G, n, k);
        total_blocks_encoded++;
    }

    if (bit_count_out > 0) {
        byte_out <<= (8 - bit_count_out);
        fwrite(&byte_out, 1, 1, fout);
    }

    printf("Encoding complete. %ld block(s) encoded.\n", total_blocks_encoded);

    fclose(fin);
    fclose(fout);
    for(int i=0; i<k; ++i) free(G[i]);
    free(G);

    return 0;
}

// --- Helper & CRC Functions ---
Tensor_i create_tensor_i(int d1, int d2, int d3) {
    Tensor_i t;
    t.dim1 = d1; t.dim2 = d2; t.dim3 = d3;
    t.data = (int*)calloc(d1 * d2 * d3, sizeof(int));
    return t;
}
void free_tensor_i(Tensor_i t) { free(t.data); }
int get_tensor_i(Tensor_i t, int i, int j, int k) { return t.data[k*t.dim1*t.dim2 + j*t.dim1 + i]; }
void set_tensor_i(Tensor_i t, int i, int j, int k, int val) { t.data[k*t.dim1*t.dim2 + j*t.dim1 + i] = val; }

void getGH_sys_CRC(int n, int k, int** G) {
    const char* hex_poly = NULL;
    int r = n - k;
    if (r == 5 && k == 10) hex_poly = "0x15";
    else { fprintf(stderr, "Error: Unsupported (n,k) pair (%d,%d).\n", n, k); exit(1); }

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
                for (int l = 0; l < poly_len; l++) msg_poly[j + l] ^= poly[l];
            }
        }
        for (int j = 0; j < r; j++) P[i][j] = msg_poly[k + j];
    }

    for (int i = 0; i < k; i++) {
        G[i][i] = 1; // Identity part
        for (int j = 0; j < r; j++) G[i][k + j] = P[i][j]; // Parity part
    }

    free(poly);
    free(msg_poly);
    for(int i=0; i<k; ++i) free(P[i]);
    free(P);
}

int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) poly[i] = (dec_val >> (len - 1 - i)) & 1;
    poly[len] = 1;
    return poly;
}
