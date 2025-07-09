#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <error_probability>\n", argv[0]);
        fprintf(stderr, "  <error_probability> is a float between 0.0 and 1.0\n");
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    double error_prob = atof(argv[3]);

    if (error_prob < 0.0 || error_prob > 1.0) {
        fprintf(stderr, "Error: Probability must be between 0.0 and 1.0\n");
        return 1;
    }

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

    printf("Simulating channel with error probability p = %f...\n", error_prob);
    srand(time(NULL));

    int byte_in;
    long long total_bits = 0;
    long long flipped_bits = 0;

    while ((byte_in = fgetc(fin)) != EOF) {
        unsigned char byte_out = 0;
        for (int i = 7; i >= 0; i--) {
            int bit = (byte_in >> i) & 1;
            total_bits++;
            if ((double)rand() / RAND_MAX < error_prob) {
                bit = 1 - bit;
                flipped_bits++;
            }
            byte_out = (byte_out << 1) | bit;
        }
        fputc(byte_out, fout);
    }

    printf("Channel simulation complete.\n");
    printf("Processed %lld bits and flipped %lld bits.\n", total_bits, flipped_bits);

    fclose(fin);
    fclose(fout);

    return 0;
}
