#include <immintrin.h>

void convert(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    // 512 bit registers, 32 bit floats => 16
    int floats_per_operation = 16;

    int size = width * height;
    int pixel_per_thread_unaligned = size / threads;
    // Each FMA instruction can calculate 16 pixels at once, so we need a worksize that is a multiple of it.
    // Leftover will need to be handled seperatly without FMA by the last thread.
    int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;

    __m512 r_factor = _mm512_set_ps(0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126, 0.2126);
    __m512 g_factor = _mm512_set_ps(0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152, 0.7152);
    __m512 b_factor = _mm512_set_ps(0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722, 0.0722);

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

        __m512 r_vector, g_vector, b_vector, gray_vector;
        __m512i gray_vector_int;
        for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
        {
            r_vector = _mm512_set_ps(img[(i * channels)], img[(i + 1) * channels], img[(i + 2) * channels], img[(i + 3) * channels], img[(i + 4) * channels], img[(i + 5) * channels], img[(i + 6) * channels], img[(i + 7) * channels], img[(i + 8) * channels], img[(i + 9) * channels], img[(i + 10) * channels], img[(i + 11) * channels], img[(i + 12) * channels], img[(i + 13) * channels], img[(i + 14) * channels], img[(i + 15) * channels]);
            g_vector = _mm512_set_ps(img[(i * channels) + 1], img[(i + 1) * channels + 1], img[(i + 2) * channels + 1], img[(i + 3) * channels + 1], img[(i + 4) * channels + 1], img[(i + 5) * channels + 1], img[(i + 6) * channels + 1], img[(i + 7) * channels + 1], img[(i + 8) * channels + 1], img[(i + 9) * channels + 1], img[(i + 10) * channels + 1], img[(i + 11) * channels + 1], img[(i + 12) * channels + 1], img[(i + 13) * channels + 1], img[(i + 14) * channels + 1], img[(i + 15) * channels + 1]);
            b_vector = _mm512_set_ps(img[(i * channels) + 2], img[(i + 1) * channels + 2], img[(i + 2) * channels + 2], img[(i + 3) * channels + 2], img[(i + 4) * channels + 2], img[(i + 5) * channels + 2], img[(i + 6) * channels + 2], img[(i + 7) * channels + 2], img[(i + 8) * channels + 2], img[(i + 9) * channels + 2], img[(i + 10) * channels + 2], img[(i + 11) * channels + 2], img[(i + 12) * channels + 2], img[(i + 13) * channels + 2], img[(i + 14) * channels + 2], img[(i + 15) * channels + 2]);

            // calculate gray value with FMA
            gray_vector = _mm512_setzero_ps();
            gray_vector = _mm512_fmadd_ps(r_vector, r_factor, gray_vector);
            gray_vector = _mm512_fmadd_ps(g_vector, g_factor, gray_vector);
            gray_vector = _mm512_fmadd_ps(b_vector, b_factor, gray_vector);

            // convert float to int and store it
            // https://stackoverflow.com/q/29856006
            gray_vector_int = _mm512_cvtps_epi32(gray_vector);
            // gray_vector_int = _mm512_packus_epi32(gray_vector_int, gray_vector_int);
            gray_vector_int = _mm512_packus_epi16(gray_vector_int, gray_vector_int);

            _mm512_storeu_si512((__m512i_u *) &result[i], gray_vector_int);
        }
    }

    // calculate the leftover pixels which result from the image not having a
    // pixel count that is a multiple of 16
    // should be 15 pixels at most
    int start = ((int)size / floats_per_operation) * floats_per_operation;
    for (int i = start; i < size; i++)
    {
        result[i] =
            0.2126 * img[(i * channels)]        // red
            + 0.7152 * img[(i * channels) + 1]  // green
            + 0.0722 * img[(i * channels) + 2]; // blue
    }
}