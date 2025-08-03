#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to generate a standard normal random number
double normal_dist_rand() {
    double u1 = (double)rand() / (RAND_MAX + 1.0);
    double u2 = (double)rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <Eb/N0_in_dB>\n", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    double EbN0dB = atof(argv[3]);

    // Calculate noise variance for BPSK modulation
    double EbN0_linear = pow(10, EbN0dB / 10.0);
    double sigma = 1.0 / sqrt(2.0 * EbN0_linear);

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

    printf("Simulating AWGN channel for Eb/N0 = %.2f dB (sigma = %f)...\n", EbN0dB, sigma);
    srand(time(NULL));

    int byte_in;
    while ((byte_in = fgetc(fin)) != EOF) {
        for (int i = 7; i >= 0; i--) {
            int bit = (byte_in >> i) & 1;

            // 1. Modulate (BPSK): 0 -> +1, 1 -> -1
            double x = (bit == 0) ? 1.0 : -1.0;

            // 2. Add noise
            double y = x + sigma * normal_dist_rand();

            // 3. Hard decision: positive -> 0, negative -> 1
            int received_bit = (y > 0) ? 0 : 1;

            // 4. Write received bit to output file
            fwrite(&received_bit, sizeof(int), 1, fout);
        }
    }

    printf("Channel simulation complete. Output file '%s' contains received bits.\n", output_filename);

    fclose(fin);
    fclose(fout);

    return 0;
}
