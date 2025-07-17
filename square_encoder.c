#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Forward Declarations ---
int** create_int_matrix(int rows, int cols);
void free_int_matrix(int** matrix, int rows);
int* koopman2matlab(const char* k_poly, int* poly_len);
void getGH_sys_CRC(int n, int k, int** G);

// --- Main Application ---
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    const int n = 31;
    const int k = 25;
    const int message_block_size = k * k; // 625 bits
    const int codeword_block_size = n * n; // 961 bits

    int** G = create_int_matrix(k, n);
    getGH_sys_CRC(n, k, G);

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("Encoding %s to %s using (n=%d, k=%d) square code...\n", input_filename, output_filename, n, k);

    int bit_buffer[message_block_size];
    int bits_in_buffer = 0;
    unsigned char byte_in;
    unsigned char byte_out = 0;
    int bit_count_out = 0;

    while (fread(&byte_in, 1, 1, fin) == 1) {
        for (int i = 7; i >= 0; i--) {
            bit_buffer[bits_in_buffer++] = (byte_in >> i) & 1;
            if (bits_in_buffer == message_block_size) {
                int** u = create_int_matrix(k, k);
                for(int r=0; r<k; ++r) for(int c=0; c<k; ++c) u[r][c] = bit_buffer[r*k + c];

                int** c_matrix = create_int_matrix(n, n);
                int temp_row[n];
                int temp_col_k[k];

                // Encode rows
                for (int row = 0; row < k; row++) {
                    memset(temp_row, 0, n * sizeof(int));
                    for (int col = 0; col < n; col++) for (int g_col = 0; g_col < k; g_col++) temp_row[col] += u[row][g_col] * G[g_col][col];
                    for (int col = 0; col < n; col++) c_matrix[row][col] = temp_row[col] % 2;
                }
                // Encode columns
                for (int col = 0; col < n; col++) {
                    for(int i=0; i<k; ++i) temp_col_k[i] = c_matrix[i][col];
                    memset(temp_row, 0, n * sizeof(int));
                    for (int row = 0; row < n; row++) for (int g_col = 0; g_col < k; g_col++) temp_row[row] += temp_col_k[g_col] * G[g_col][row];
                    for (int row = 0; row < n; row++) c_matrix[row][col] = temp_row[row] % 2;
                }

                for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) {
                    byte_out = (byte_out << 1) | c_matrix[r][c];
                    if (++bit_count_out == 8) {
                        fwrite(&byte_out, 1, 1, fout);
                        byte_out = 0;
                        bit_count_out = 0;
                    }
                }
                free_int_matrix(u, k);
                free_int_matrix(c_matrix, n);
                bits_in_buffer = 0;
            }
        }
    }
    if (bits_in_buffer > 0) {
        for (int i = bits_in_buffer; i < message_block_size; i++) bit_buffer[i] = 0;
        int** u = create_int_matrix(k, k);
        for(int r=0; r<k; ++r) for(int c=0; c<k; ++c) u[r][c] = bit_buffer[r*k + c];
        int** c_matrix = create_int_matrix(n, n);
        int temp_row[n];
        int temp_col_k[k];
        for (int row = 0; row < k; row++) { memset(temp_row, 0, n * sizeof(int)); for (int col = 0; col < n; col++) for (int g_col = 0; g_col < k; g_col++) temp_row[col] += u[row][g_col] * G[g_col][col]; for (int col = 0; col < n; col++) c_matrix[row][col] = temp_row[col] % 2; }
        for (int col = 0; col < n; col++) { for(int i=0; i<k; ++i) temp_col_k[i] = c_matrix[i][col]; memset(temp_row, 0, n * sizeof(int)); for (int row = 0; row < n; row++) for (int g_col = 0; g_col < k; g_col++) temp_row[row] += temp_col_k[g_col] * G[g_col][row]; for (int row = 0; row < n; row++) c_matrix[row][col] = temp_row[row] % 2; }
        for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) {
            byte_out = (byte_out << 1) | c_matrix[r][c];
            if (++bit_count_out == 8) { fwrite(&byte_out, 1, 1, fout); byte_out = 0; bit_count_out = 0; }
        }
        free_int_matrix(u, k);
        free_int_matrix(c_matrix, n);
    }
    if (bit_count_out > 0) { byte_out <<= (8 - bit_count_out); fwrite(&byte_out, 1, 1, fout); }

    printf("Encoding complete.\n");
    fclose(fin); fclose(fout);
    for(int i=0; i<k; ++i) free(G[i]);
    free(G);
    return 0;
}
int** create_int_matrix(int r, int c) { int** m = (int**)malloc(r * sizeof(int*)); for (int i = 0; i < r; i++) m[i] = (int*)calloc(c, sizeof(int)); return m; }
void free_int_matrix(int** m, int r) { for (int i = 0; i < r; i++) free(m[i]); free(m); }
int* koopman2matlab(const char* p, int* l) { long long v = strtoll(p, NULL, 16); int len = (v > 0) ? floor(log2(v)) + 1 : 1; *l = len + 1; int* poly = (int*)malloc(sizeof(int) * (*l)); for (int i=0; i<len; i++) poly[i] = (v >> (len-1-i)) & 1; poly[len] = 1; return poly; }
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
