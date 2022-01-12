#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

// 32 byte alignement, SIMD load&store commands need them
// all values need to be defined if any of them is defined
#define STBI_MALLOC(sz)  aligned_alloc(32, sz)
#define STBI_REALLOC(p, newsz) realloc(p, newsz)
#define STBI_FREE(p) free(p)

#define STB_IMAGE_IMPLEMENTATION
#include "../baseline/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../baseline/stb_image_write.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "algorithms/openmp_baseline.c"
#include "algorithms/memory.c"
#include "algorithms/memory_simd_fma.c"
#include "algorithms/memory_simd_sse.c"
#include "algorithms/memory_simd_avx.c"

int main(int argc, char *argv[])
{
    int runs;
    int threads;
    int write_image;
    int algo;
    if (argc == 5)
    {
        runs = atoi(argv[1]);
        threads = atoi(argv[2]);
        write_image = atoi(argv[3]);
        algo = atoi(argv[4]);
    }
    else
    {
        printf("cli arguments: runs threads write_image algorithm_name\n");
        printf("algorithms are: 1-baseline, 2-memory, 3-fma, 4-sse, 5-avx\n");
        exit(1);
    }

    // Read color JPG into byte array "img"
    // Array contains "width" x "height" pixels each consisting of "channels" colors/bytes
    int width, height, channels;
    //https://photojournal.jpl.nasa.gov/catalog/?IDNumber=PIA03239
    unsigned char *img = stbi_load("../images/27000x6000.jpg", &width, &height, &channels, 0);
    if (img == NULL)
    {
        printf("Err: loading image\n");
        exit(1);
    }

    printf("w: %d ; h: %d ; c: %d ;\n", width, height, channels, runs, threads);
    printf("algo: %d; runs: %d ; threads: %d\n", algo, runs, threads);

    // Allocate target array for grayscale image
    unsigned char *gray = aligned_alloc(32, width * height * sizeof(char));
    if (gray == NULL)
    {
        printf("Err: allocating gray image\n");
        exit(1);
    }
    double time_sum = 0.0;
    omp_set_num_threads(threads);

    for (int i = 0; i < runs; i++)
    {
        // start time tracking
        struct timeval start;
        gettimeofday(&start, 0);

        // convert
        switch (algo)
        {
        case 1:
            convert_baseline(img, width, height, channels, threads, gray);
            break;
        case 2:
            convert_memory(img, width, height, channels, threads, gray);
            break;
        case 3:
            convert_memory_simd_fma(img, width, height, channels, threads, gray);
            break;
        case 4:
            convert_memory_simd_sse(img, width, height, channels, threads, gray);
            break;
        case 5:
            convert_memory_simd_avx(img, width, height, channels, threads, gray);
            break;
        default:
            printf("Unknown algorithm\n");
            printf("algorithms are: 1-baseline, 2-memory, 3-fma, 4-sse, 5-avx\n");
            exit(1);
            break;
        }

        // end time tracking
        struct timeval end;
        gettimeofday(&end, 0);
        long lsec = end.tv_sec - start.tv_sec;
        long lusec = end.tv_usec - start.tv_usec;
        double sec = (lsec + lusec / 1000000.0);
        time_sum += sec;
        printf("%8.6f seconds\n", sec);

        // add a bit of a pause for the CPU between runs
        sleep(0.5);
    }
    printf("average: %8.6f second, %8.4f mega pixel per second\nFor Markdown: \n", time_sum / runs, (width * height) / (time_sum * 1024 * 1024 / runs));
    printf("|%8.6f|%8.4f|\n", time_sum / runs, (width * height) / (time_sum * 1024 * 1024 / runs));

    if (write_image == 1)
    {
        printf("Writing image\n");
        stbi_write_jpg("grayscale.jpg", width, height, 1, gray, 95);
    }
}