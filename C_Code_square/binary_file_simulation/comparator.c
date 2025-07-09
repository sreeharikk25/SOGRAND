#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <original_file> <decoded_file>\n", argv[0]);
        return 1;
    }

    const char* original_filename = argv[1];
    const char* decoded_filename = argv[2];

    FILE* f_orig = fopen(original_filename, "rb");
    if (!f_orig) {
        perror("Error opening original file");
        return 1;
    }
    FILE* f_dec = fopen(decoded_filename, "rb");
    if (!f_dec) {
        perror("Error opening decoded file");
        fclose(f_orig);
        return 1;
    }

    printf("Comparing %s and %s...\n", original_filename, decoded_filename);

    long long total_bits = 0;
    long long error_bits = 0;
    int byte_orig, byte_dec;

    while ((byte_orig = fgetc(f_orig)) != EOF) {
        byte_dec = fgetc(f_dec);
        if (byte_dec == EOF) {
            printf("Error: Decoded file is shorter than original file.\n");
            error_bits += 8; // Count the whole missing byte as errors
            total_bits += 8;
            break;
        }

        // Compare bit by bit
        for (int i = 7; i >= 0; i--) {
            int bit_orig = (byte_orig >> i) & 1;
            int bit_dec = (byte_dec >> i) & 1;
            if (bit_orig != bit_dec) {
                error_bits++;
            }
            total_bits++;
        }
    }

    printf("\n--- Comparison Report ---\n");
    if (error_bits == 0) {
        printf("SUCCESS: Files are identical.\n");
    } else {
        printf("FAILURE: Files are different.\n");
    }
    printf("Total bits compared: %lld\n", total_bits);
    printf("Total bit errors:    %lld\n", error_bits);
    if (total_bits > 0) {
        printf("Bit Error Rate (BER): %e\n", (double)error_bits / total_bits);
    }

    fclose(f_orig);
    fclose(f_dec);

    return 0;
}
