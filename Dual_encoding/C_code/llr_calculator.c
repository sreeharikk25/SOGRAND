#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <received_bits_file> <llr_output_file> <Eb/N0_in_dB> <code_dimension>\n", argv[0]);
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

    // --- Code parameters for calculating noise variance ---
    const int n = 15;
    const int k = 10;
    const double R = pow((double)k / n, code_dim); // Rate depends on dimension

    // --- Calculate sigma from Eb/N0 ---
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

    printf("Converting received bits to LLRs for Eb/N0 = %.2f dB (sigma = %f)...\n", EbN0dB, sigma);

    int received_bit;
    while (fread(&received_bit, sizeof(int), 1, fin) == 1) {
        // Convert received bit back to received symbol
        // 0 -> +1, 1 -> -1 (reverse of modulation)
        double y = (received_bit == 0) ? 1.0 : -1.0;

        // Add some uncertainty based on noise level
        // This simulates the soft information we would have
        // In practice, you'd want the actual received analog values
        // For now, we'll use a simplified approach

        // Calculate LLR: LLR = 2*y/sigma^2
        double llr = 2.0 * y / variance;

        // Write LLR to output file
        fwrite(&llr, sizeof(double), 1, fout);
    }

    printf("LLR calculation complete. Output file '%s' contains LLRs.\n", output_filename);

    fclose(fin);
    fclose(fout);

    return 0;
}
