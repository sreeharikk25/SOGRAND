#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    const char *filename = "original_data.bin";
    const int num_bytes = 8*8*8*1000/8;
    // was 625 - Note: 10 blocks * 1000 bits/block / 8 bits/byte

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    srand(time(NULL));
    printf("Generating %d random bytes into %s...\n", num_bytes, filename);

    for (int i = 0; i < num_bytes; i++) {
        unsigned char byte = rand() % 256;
        fwrite(&byte, sizeof(char), 1, fp);
    }

    fclose(fp);
    printf("Successfully created binary file: %s\n", filename);
    return 0;
}
