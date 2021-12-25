#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../baseline/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../baseline/stb_image_write.h"

#define THREADS 8

void convert_openmp_baseline(unsigned char *img, int width, int height, int channels, unsigned char *result)
{
#pragma omp parallel for collapse(2)
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            result[y * width + x] =
                0.2126 * img[(y * width + x) * channels]        // red
                + 0.7152 * img[(y * width + x) * channels + 1]  // green
                + 0.0722 * img[(y * width + x) * channels + 2]; // blue
        }
    }
}

void convert_openmp_memory(unsigned char *img, int width, int height, int channels, unsigned char *result)
{
    int pixel_per_thread = (width * height) / THREADS;
#pragma omp parallel for
    for (int thread = 0; thread < THREADS; thread++)
    {
        int end;
        if (thread + 1 == THREADS)
        {
            end = width * height;
        }
        else
        {
            end = pixel_per_thread * (thread + 1);
        }

        for (int i = pixel_per_thread * thread; i < end; i++)
        {
            result[i] =
                0.2126 * img[(i * channels) + 1]    // red
                + 0.7152 * img[(i * channels) + 2]  // green
                + 0.0722 * img[(i * channels) + 3]; // blue
        }
    }
}

void convert_openmp_memory_simd(unsigned char *img, int width, int height, int channels, unsigned char *result)
{
    int pixel_per_thread = (width * height) / THREADS;
#pragma omp parallel for
    for (int thread = 0; thread < THREADS; thread++)
    {
        int end;
        if (thread + 1 == THREADS)
        {
            end = width * height;
        }
        else
        {
            end = pixel_per_thread * (thread + 1);
        }

        float *gray_pixel_values = malloc(4*sizeof(float));
        __m128 factors = _mm_setr_ps(0.2126, 0.7152, 0.0722, 0);

        for (int i = pixel_per_thread * thread; i < end; i++)
        {
            __m128 pixel = _mm_setr_ps((float)img[(i * channels) + 1], (float)img[(i * channels) + 2], (float)img[(i * channels) + 3], 0);
            __m128 gray_pixel_values_vector = _mm_mul_ps(pixel, factors);
            _mm_store_ps(gray_pixel_values, gray_pixel_values_vector);

            result[i] = gray_pixel_values[0] * gray_pixel_values[1] * gray_pixel_values[2];
        }
        free(gray_pixel_values);
    }
}

int main()
{
    int runs = 20;

    // Read color JPG into byte array "img"
    // Array contains "width" x "height" pixels each consisting of "channels" colors/bytes
    int width, height, channels;
    // unsigned char *img = stbi_load("../images/15360x8640.jpg", &width, &height, &channels, 0);
    unsigned char *img = stbi_load("../images/7680x4320.jpg", &width, &height, &channels, 0);
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
        // convert_openmp_baseline(img, width, height, channels, gray);
        // convert_openmp_memory(img, width, height, channels, gray);
        convert_openmp_memory_simd(img, width, height, channels, gray);

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