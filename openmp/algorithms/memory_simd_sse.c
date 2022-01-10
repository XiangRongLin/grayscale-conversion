#include <immintrin.h>
#include <stdio.h>

// From https://stackoverflow.com/a/57844027

//Convert from RGBRGBRGB... to RRR..., GGG..., BBB...
//Input: Two XMM registers (24 uint8 elements) ordered RGBRGB...
//Output: Three XMM registers ordered RRR..., GGG... and BBB...
//        Unpack the result from uint8 elements to uint16 elements.
static __attribute__((always_inline)) inline void GatherRGBx8(const __m128i r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
                                                              const __m128i b7_g7_r7_b6_g6_r6_b5_g5,
                                                              __m128i *r7_r6_r5_r4_r3_r2_r1_r0,
                                                              __m128i *g7_g6_g5_g4_g3_g2_g1_g0,
                                                              __m128i *b7_b6_b5_b4_b3_b2_b1_b0)
{
    //Shuffle mask for gathering 4 R elements, 4 G elements and 4 B elements (also set last 4 elements to duplication of first 4 elements).
    const __m128i shuffle_mask = _mm_set_epi8(9, 6, 3, 0, 11, 8, 5, 2, 10, 7, 4, 1, 9, 6, 3, 0);

    __m128i b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4 = _mm_alignr_epi8(b7_g7_r7_b6_g6_r6_b5_g5, r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0, 12);

    //Gather 4 R elements, 4 G elements and 4 B elements.
    //Remark: As I recall _mm_shuffle_epi8 instruction is not so efficient (I think execution is about 5 times longer than other shuffle instructions).
    __m128i r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0 = _mm_shuffle_epi8(r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0, shuffle_mask);
    __m128i r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4 = _mm_shuffle_epi8(b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4, shuffle_mask);

    //Put 8 R elements in lower part.
    __m128i b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4_r3_r2_r1_r0 = _mm_alignr_epi8(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4, r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0, 12);

    //Put 8 G elements in lower part.
    __m128i g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz_zz_zz_zz_zz = _mm_slli_si128(r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0, 8);
    __m128i zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4 = _mm_srli_si128(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4, 4);
    __m128i r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_g3_g2_g1_g0 = _mm_alignr_epi8(zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4, g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz_zz_zz_zz_zz, 12);

    //Put 8 B elements in lower part.
    __m128i b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz = _mm_slli_si128(r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0, 4);
    __m128i zz_zz_zz_zz_zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4 = _mm_srli_si128(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4, 8);
    __m128i zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_b3_b2_b1_b0 = _mm_alignr_epi8(zz_zz_zz_zz_zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4, b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz, 12);

    //Unpack uint8 elements to uint16 elements.
    *r7_r6_r5_r4_r3_r2_r1_r0 = _mm_cvtepu8_epi16(b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4_r3_r2_r1_r0);
    *g7_g6_g5_g4_g3_g2_g1_g0 = _mm_cvtepu8_epi16(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_g3_g2_g1_g0);
    *b7_b6_b5_b4_b3_b2_b1_b0 = _mm_cvtepu8_epi16(zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_b3_b2_b1_b0);
}

//Calculate 8 Grayscale elements from 8 RGB elements.
//Y = 0.2989*R + 0.5870*G + 0.1140*B
//Conversion model used by MATLAB https://www.mathworks.com/help/matlab/ref/rgb2gray.html
static __attribute__((always_inline)) inline __m128i Rgb2Yx8(__m128i r7_r6_r5_r4_r3_r2_r1_r0,
                                                             __m128i g7_g6_g5_g4_g3_g2_g1_g0,
                                                             __m128i b7_b6_b5_b4_b3_b2_b1_b0)
{
    //Each coefficient is expanded by 2^15, and rounded to int16 (add 0.5 for rounding).
    const __m128i r_coef = _mm_set1_epi16((short)(0.2126 * 32768.0 + 0.5)); //8 coefficients - R scale factor.
    const __m128i g_coef = _mm_set1_epi16((short)(0.7152 * 32768.0 + 0.5)); //8 coefficients - G scale factor.
    const __m128i b_coef = _mm_set1_epi16((short)(0.0722 * 32768.0 + 0.5)); //8 coefficients - B scale factor.

    //Multiply input elements by 64 for improved accuracy.
    r7_r6_r5_r4_r3_r2_r1_r0 = _mm_slli_epi16(r7_r6_r5_r4_r3_r2_r1_r0, 6);
    g7_g6_g5_g4_g3_g2_g1_g0 = _mm_slli_epi16(g7_g6_g5_g4_g3_g2_g1_g0, 6);
    b7_b6_b5_b4_b3_b2_b1_b0 = _mm_slli_epi16(b7_b6_b5_b4_b3_b2_b1_b0, 6);

    //Use the special intrinsic _mm_mulhrs_epi16 that calculates round(r*r_coef/2^15).
    //Calculate Y = 0.2989*R + 0.5870*G + 0.1140*B (use fixed point computations)
    __m128i y7_y6_y5_y4_y3_y2_y1_y0 = _mm_add_epi16(_mm_add_epi16(
                                                        _mm_mulhrs_epi16(r7_r6_r5_r4_r3_r2_r1_r0, r_coef),
                                                        _mm_mulhrs_epi16(g7_g6_g5_g4_g3_g2_g1_g0, g_coef)),
                                                    _mm_mulhrs_epi16(b7_b6_b5_b4_b3_b2_b1_b0, b_coef));

    //Divide result by 64.
    y7_y6_y5_y4_y3_y2_y1_y0 = _mm_srli_epi16(y7_y6_y5_y4_y3_y2_y1_y0, 6);

    return y7_y6_y5_y4_y3_y2_y1_y0;
}

void convert(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    // 256 bit registers, 32 bit floats => 8
    int floats_per_operation = 8;

    int size = width * height;
    int pixel_per_thread_unaligned = size / threads;
    // Each FMA instruction can calculate 8 pixels at once, so we need a worksize that is a multiple of it.
    // Leftover will need to be handled seperatly without FMA by the last thread.
    int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;

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

        __m128i r7_r6_r5_r4_r3_r2_r1_r0;
        __m128i g7_g6_g5_g4_g3_g2_g1_g0;
        __m128i b7_b6_b5_b4_b3_b2_b1_b0;



        for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
        {
            __m128i r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0 = _mm_loadu_si128((__m128i *)&img[i * channels]); //Unaligned load of 16 uint8 elements
            __m128i b7_g7_r7_b6_g6_r6_b5_g5 = _mm_loadu_si128((__m128i *)&img[i * channels + 16]);                    //Unaligned load of (only) 8 uint8 elements (lower half of XMM register).

            //Separate RGB, and put together R elements, G elements and B elements (together in same XMM register).
            //Result is also unpacked from uint8 to uint16 elements.
            GatherRGBx8(r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
                        b7_g7_r7_b6_g6_r6_b5_g5,
                        &r7_r6_r5_r4_r3_r2_r1_r0,
                        &g7_g6_g5_g4_g3_g2_g1_g0,
                        &b7_b6_b5_b4_b3_b2_b1_b0);

            //Calculate 8 Y elements.
            __m128i y7_y6_y5_y4_y3_y2_y1_y0 = Rgb2Yx8(r7_r6_r5_r4_r3_r2_r1_r0,
                                                      g7_g6_g5_g4_g3_g2_g1_g0,
                                                      b7_b6_b5_b4_b3_b2_b1_b0);

            //Pack uint16 elements to 16 uint8 elements (put result in single XMM register). Only lower 8 uint8 elements are relevant.
            __m128i j7_j6_j5_j4_j3_j2_j1_j0 = _mm_packus_epi16(y7_y6_y5_y4_y3_y2_y1_y0, y7_y6_y5_y4_y3_y2_y1_y0);

            //Store 8 elements of Y in row Y0, and 8 elements of Y in row Y1.
            _mm_storel_epi64((__m128i *)&result[i], j7_j6_j5_j4_j3_j2_j1_j0);
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
