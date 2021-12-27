#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../baseline/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../baseline/stb_image_write.h"

// Comment in whichever algorithm should be used and comment out all the other ones.
// #include "algorithms/baseline.c"
// #include "algorithms/memory.c"
// #include "algorithms/memory_simd.c"
// #include "algorithms/memory_simd_fma.c"
// #include "algorithms/memory_simd_fma2.c"
#include "algorithms/memory_simd_fma_256_bit.c"
// #include "algorithms/memory_simd_256_bit.c"
// #include "algorithms/memory_simd_fma_512_bit.c"

#define THREADS 8

int main()
{
    int runs = 20;

    // Read color JPG into byte array "img"
    // Array contains "width" x "height" pixels each consisting of "channels" colors/bytes
    int width, height, channels;
    unsigned char *img = stbi_load("../images/15360x8640.jpg", &width, &height, &channels, 0);
    // unsigned char *img = stbi_load("../images/7680x4320.jpg", &width, &height, &channels, 0);
    if (img == NULL)
    {
        printf("Err: loading image\n");
        exit(1);
    }

    printf("w: %d ; h: %d ; c: %d\n", width, height, channels);

    // Allocate target array for grayscale image
    unsigned char *gray = malloc(width * height);
    double mflops_sum = 0.0;
    double time_sum = 0.0;

    for (int i = 0; i < runs; i++)
    {
        // start time tracking
        struct timeval start;
        gettimeofday(&start, 0);

        // convert
        omp_set_num_threads(THREADS);
        convert(img, width, height, channels, THREADS, gray);

        // end time tracking
        struct timeval end;
        gettimeofday(&end, 0);
        long lsec = end.tv_sec - start.tv_sec;
        long lusec = end.tv_usec - start.tv_usec;
        double sec = (lsec + lusec / 1000000.0);
        printf("%8.6f seconds\n", sec);

        // FLOP calculation
        double flop = width * height * (3 + 2);
        printf("%8.2f MFLOP\n", flop / 1000000.0);

        // Print FLOP/s
        double mflops = flop / 1000000.0 / sec;
        mflops_sum += mflops;
        time_sum += sec;
        printf("%8.2f MFLOP/s\n", mflops);
    }

    printf("average: %8.2f MFLOP/s - %8.6f seconds\n", mflops_sum / runs, time_sum / runs);

    // printf("Writing image\n");
    // stbi_write_jpg("grayscale.jpg", width, height, 1, gray, 95);
}