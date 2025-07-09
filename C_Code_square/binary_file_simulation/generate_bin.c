#include <stdio.h>
#include <string.h>

int main() {
    const char *filename = "original_data.bin";
    const char *data = "Hello, world!";

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    fwrite(data, sizeof(char), strlen(data), fp);

    fclose(fp);

    printf("Successfully created binary file: %s\n", filename);

    return 0;
}
