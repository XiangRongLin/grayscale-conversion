#include <immintrin.h>
#include <stdio.h>

const uint16_t r_const = 0.2126 * 0x10000;
const uint16_t g_const = 0.7152 * 0x10000;
const uint16_t b_const = 0.0722 * 0x10000;

// From https://github.com/Const-me/SimdIntroArticle/tree/master/Grayscale
// NOT WORKING

// Pack red channel of 16 pixels into uint16_t lanes, in [ 0 .. 0xFF00 ] interval.
// The order of the pixels is a0, a1, a2, a3, b0, b1, b2, b3, a4, a5, a6, a7, b4, b5, b6, b7.
inline __attribute__((always_inline)) __m256i packRed(__m256i a, __m256i b)
{
    const __m256i mask = _mm256_set1_epi32(0xFF);
    a = _mm256_and_si256(a, mask);
    b = _mm256_and_si256(b, mask);
    const __m256i packed = _mm256_packus_epi32(a, b);
    return _mm256_slli_si256(packed, 1);
}

// Pack green channel of 16 pixels into uint16_t lanes, in [ 0 .. 0xFF00 ] interval
inline __attribute__((always_inline)) __m256i packGreen(__m256i a, __m256i b)
{
    const __m256i mask = _mm256_set1_epi32(0xFF00);
    a = _mm256_and_si256(a, mask);
    b = _mm256_and_si256(b, mask);
    return _mm256_packus_epi32(a, b);
}

// Pack blue channel of 16 pixels into uint16_t lanes, in [ 0 .. 0xFF00 ] interval
inline __attribute__((always_inline)) __m256i packBlue(__m256i a, __m256i b)
{
    const __m256i mask = _mm256_set1_epi32(0xFF00);
    a = _mm256_srli_si256(a, 1);
    b = _mm256_srli_si256(b, 1);
    a = _mm256_and_si256(a, mask);
    b = _mm256_and_si256(b, mask);
    return _mm256_packus_epi32(a, b);
}

// Load 16 pixels, split into RGB channels
inline __attribute__((always_inline)) void loadRgb(const __m256i_u *src, __m256i *red, __m256i *green, __m256i *blue)
{
    const __m256i a = _mm256_loadu_si256(src);
    const __m256i b = _mm256_loadu_si256(src + 1);
    *red = packRed(a, b);
    *green = packGreen(a, b);
    *blue = packBlue(a, b);
}

// Compute brightness of 16 pixels. Input is 16-bit numbers in [ 0 .. 0xFF00 ] interval, output is 16-bit numbers in [ 0 .. 0xFF ] interval.
inline __attribute__((always_inline)) __m256i brightness(__m256i r, __m256i g, __m256i b)
{
    r = _mm256_mulhi_epu16(r, _mm256_set1_epi16((short)r_const));
    g = _mm256_mulhi_epu16(g, _mm256_set1_epi16((short)g_const));
    b = _mm256_mulhi_epu16(b, _mm256_set1_epi16((short)b_const));
    const __m256i result = _mm256_adds_epu16(_mm256_adds_epu16(r, g), b);
    return _mm256_srli_epi16(result, 8);
}

void convert(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    // 256 bit registers, 32 bit floats => 8
    const int floats_per_operation = 8;

    const int size = width * height;
    const int pixel_per_thread_unaligned = size / threads;
    // Each FMA instruction can calculate 8 pixels at once, so we need a worksize that is a multiple of it.
    // Leftover will need to be handled seperatly without FMA by the last thread.
    const int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;

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

        __m256i r_vector, g_vector, b_vector, gray_vector;
        __m256i gray_vector_int;
        for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
        {
            // Compute brightness of 32 pixels.
            loadRgb((__m256i_u *)&img[(i * channels)], &r_vector, &g_vector, &b_vector);
            __m256i low = brightness(r_vector, g_vector, b_vector);
            loadRgb((__m256i_u *)&img[((i + floats_per_operation) * channels)], &r_vector, &g_vector, &b_vector);
            __m256i hi = brightness(r_vector, g_vector, b_vector);

            // The pixel order is weird in low/high variables, due to the way 256-bit AVX2 pack instructions are implemented. They both contain pixels in the following order:
            // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7,  12, 13, 14, 15
            // Permute them to be sequential by shuffling 64-bit blocks.
            // constexpr int permuteControl = _MM_SHUFFLE( 3, 1, 2, 0 );
            low = _mm256_permute4x64_epi64(low, 216);
            hi = _mm256_permute4x64_epi64(hi, 216);

            // Pack 16-bit integers into bytes
            __m256i bytes = _mm256_packus_epi16(low, hi);

            // Once again, fix the order after 256-bit pack instruction.
            bytes = _mm256_permute4x64_epi64(bytes, 216);

            // Store the results
            _mm256_storeu_si256((__m256i_u *)result, bytes);
        }
    }

    // calculate the leftover pixels which result from the image not having a
    // pixel count that is a multiple of 8
    // should be 7 pixels at most
    const int start = ((int)size / floats_per_operation) * floats_per_operation;
    for (int i = start; i < size; i++)
    {
        result[i] =
            0.2126 * img[(i * channels)]        // red
            + 0.7152 * img[(i * channels) + 1]  // green
            + 0.0722 * img[(i * channels) + 2]; // blue
    }
}