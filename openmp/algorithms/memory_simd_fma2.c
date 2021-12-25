#include <immintrin.h>

void convert_openmp_memory_simd_fma2(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    int size = width * height;
    int pixel_per_thread_unaligned = size / threads;
    // Each FMA instruction can calculate 4 pixels at once, so we need a worksize that is a multiple of it.
    // Leftover will need to be handled seperatly without FMA by the last thread.
    int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / 4) * 4;

    __m128 r_factor = _mm_set_ps(0.2126, 0.2126, 0.2126, 0.2126);
    __m128 g_factor = _mm_set_ps(0.7152, 0.7152, 0.7152, 0.7152);
    __m128 b_factor = _mm_set_ps(0.0722, 0.0722, 0.0722, 0.0722);

#pragma omp parallel for
    for (int thread = 0; thread < threads; thread++)
    {
        int end;
        if (thread + 1 == threads)
        {
            end = ((int)size / 4) * 4;
        }
        else
        {
            end = pixel_per_thread_aligned * (thread + 1);
        }

        __m128 r_vector, g_vector, b_vector, gray_vector;
        __m128i gray_vector_int;
        for (int i = pixel_per_thread_aligned * thread; i < end; i += 4)
        {
            r_vector = _mm_set_ps(img[(i * channels)], img[(i * channels) + 1], img[(i * channels) + 2], img[(i * channels) + 3]);
            g_vector = _mm_set_ps(img[(i * channels)], img[(i * channels) + 1], img[(i * channels) + 2], img[(i * channels) + 3]);
            b_vector = _mm_set_ps(img[(i * channels)], img[(i * channels) + 1], img[(i * channels) + 2], img[(i * channels) + 3]);

            // calculate gray value with FMA
            gray_vector = _mm_setzero_ps();
            gray_vector = _mm_fmadd_ps(r_vector, r_factor, gray_vector);
            gray_vector = _mm_fmadd_ps(g_vector, g_factor, gray_vector);
            gray_vector = _mm_fmadd_ps(b_vector, b_factor, gray_vector);

            // convert float to int and store it
            // https://stackoverflow.com/q/29856006
            gray_vector_int = _mm_cvtps_epi32(gray_vector);
            gray_vector_int = _mm_packus_epi32(gray_vector_int, gray_vector_int);
            gray_vector_int = _mm_packus_epi16(gray_vector_int, gray_vector_int);

            *(int *)(&result[i]) = _mm_cvtsi128_si32(gray_vector_int);
        }
    }

    // calculate the leftover pixels which result from the image not having a
    // pixel count that is a multiple of 4
    // should be 3 pixels at most
    int start = ((int)size / 4) * 4;
    for (int i = start; i < size; i++)
    {
        result[i] =
            0.2126 * img[(i * channels)]        // red
            + 0.7152 * img[(i * channels) + 1]  // green
            + 0.0722 * img[(i * channels) + 2]; // blue
    }
}