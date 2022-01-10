#include <immintrin.h>
#include <stdio.h>

void convert_simd_fma_256_bit(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    // 256 bit registers, 32 bit floats => 8
    int floats_per_operation = 8;

    int size = width * height;
    int pixel_per_thread_unaligned = size / threads;
    // Each FMA instruction can calculate 8 pixels at once, so we need a worksize that is a multiple of it.
    // Leftover will need to be handled seperatly without FMA by the last thread.
    int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;

    __m256 r_factor = _mm256_set_ps(0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126);
    __m256 g_factor = _mm256_set_ps(0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152);
    __m256 b_factor = _mm256_set_ps(0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722);

#pragma omp parallel for
    for (int thread = 0; thread < threads; thread++)
    {
        int end;
        if (thread + 1 == threads)
        {
            end = ((int)size / floats_per_operation) * floats_per_operation;
        }
        else
        {
            end = pixel_per_thread_aligned * (thread + 1);
        }

        __m256 r_vector, g_vector, b_vector, gray_vector;
        __m256i gray_vector_int;
        for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
        {
            r_vector = _mm256_set_ps(img[(i * channels)], img[(i + 1) * channels], img[(i + 2) * channels], img[(i + 3) * channels], img[(i + 4) * channels], img[(i + 5) * channels], img[(i + 6) * channels], img[(i + 7) * channels]);
            g_vector = _mm256_set_ps(img[(i * channels) + 1], img[(i + 1) * channels + 1], img[(i + 2) * channels + 1], img[(i + 3) * channels + 1], img[(i + 4) * channels + 1], img[(i + 5) * channels + 1], img[(i + 6) * channels + 1], img[(i + 7) * channels + 1]);
            b_vector = _mm256_set_ps(img[(i * channels) + 2], img[(i + 1) * channels + 2], img[(i + 2) * channels + 2], img[(i + 3) * channels + 2], img[(i + 4) * channels + 2], img[(i + 5) * channels + 2], img[(i + 6) * channels + 2], img[(i + 7) * channels + 2]);

            // calculate gray value with FMA
            gray_vector = _mm256_setzero_ps();
            gray_vector = _mm256_fmadd_ps(r_vector, r_factor, gray_vector);
            gray_vector = _mm256_fmadd_ps(g_vector, g_factor, gray_vector);
            gray_vector = _mm256_fmadd_ps(b_vector, b_factor, gray_vector);

            // convert float to int and store it
            // https://stackoverflow.com/q/29856006
            gray_vector_int = _mm256_cvtps_epi32(gray_vector);
            gray_vector_int = _mm256_packus_epi32(gray_vector_int, gray_vector_int);
            gray_vector_int = _mm256_packus_epi16(gray_vector_int, gray_vector_int);

            _mm256_storeu_si256((__m256i_u *)&result[i], gray_vector_int);
        }
    }

    // calculate the leftover pixels which result from the image not having a
    // pixel count that is a multiple of 8
    // should be 7 pixels at most
    int start = ((int)size / floats_per_operation) * floats_per_operation;
    for (int i = start; i < size; i++)
    {
        result[i] =
            0.2126 * img[(i * channels)]        // red
            + 0.7152 * img[(i * channels) + 1]  // green
            + 0.0722 * img[(i * channels) + 2]; // blue
    }
}