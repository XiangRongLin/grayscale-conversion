#include <immintrin.h>

void convert_openmp_memory_simd(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    int pixel_per_thread = (width * height) / threads;
#pragma omp parallel for
    for (int thread = 0; thread < threads; thread++)
    {
        int end;
        if (thread + 1 == threads)
        {
            end = width * height;
        }
        else
        {
            end = pixel_per_thread * (thread + 1);
        }

        float *gray_pixel_values = malloc(4 * sizeof(float));
        __m128 factors = _mm_setr_ps(0.2126, 0.7152, 0.0722, 0);

        for (int i = pixel_per_thread * thread; i < end; i++)
        {
            __m128 pixel = _mm_setr_ps((float)img[(i * channels)], (float)img[(i * channels) + 1], (float)img[(i * channels) + 2], 0);
            __m128 gray_pixel_values_vector = _mm_mul_ps(pixel, factors);
            _mm_store_ps(gray_pixel_values, gray_pixel_values_vector);

            result[i] = gray_pixel_values[0] * gray_pixel_values[1] * gray_pixel_values[2];
        }
        free(gray_pixel_values);
    }
}