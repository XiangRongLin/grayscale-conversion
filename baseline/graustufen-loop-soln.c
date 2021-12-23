#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main()
{
    // Read color JPG into byte array "img"
    // Array contains "width" x "height" pixels each consisting of "channels" colors/bytes
    int width, height, channels;
    unsigned char *img = stbi_load("tamanna-rumee-vaTsR-ghLog-unsplash.jpg", &width, &height, &channels, 0);
    if (img == NULL)
    {
        printf("Err: loading image\n");
        exit(1);
    }

    printf("w: %d ; h: %d ; c: %d\n", width, height, channels);
    
    // Allocate target array for grayscale image
    unsigned char *gray = malloc(width * height);

    // TODO Zeitmessen Start
    struct timeval start;
    gettimeofday(&start, 0);

    // TODO Konvertierung
    for(int x=0;x<width;x++) {
        for(int y=0;y<height;y++) {
            gray[y * width + x] =  0.2126 * img[(y * width + x) * 3]
                                 + 0.7152 * img[(y * width + x) * 3 + 1] 
                                 + 0.0722 * img[(y * width + x) * 3 + 2];
        }
    }

    // TODO Zeitmessen Ende
    struct timeval end;
    gettimeofday(&end, 0);
    long lsec = end.tv_sec - start.tv_sec;
    long lusec = end.tv_usec - start.tv_usec;
    double sec = (lsec + lusec / 1000000.0);
    printf("%8.6f seconds\n", sec);

    // TODO FLOP Berechnung
    double flop = width * height * (3 + 2);
    printf("%8.2f MFLOP\n", flop/1000000.0);

    //TODO Ausgabe FLOP/s
    double mflops = flop / 1000000.0 / sec;
    printf("%8.2f MFLOP/s\n", mflops);


    stbi_write_jpg("grayscale.jpg", width, height, 1, gray, 95);
}