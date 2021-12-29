#include <immintrin.h>
#include <stdio.h>

//Convert from RGBRGBRGB... to RRR..., GGG..., BBB...
//Input: Two XMM registers (24 uint8 elements) ordered RGBRGB...
//Output: Three XMM registers ordered RRR..., GGG... and BBB...
//        Unpack the result from uint8 elements to uint16 elements.
static __attribute__((always_inline)) inline void GatherRGBx8(
    const __m256i gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
    const __m256i bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
    __m256i *rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
    __m256i *gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
    __m256i *bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0)
{
    // lane split between g4 and g5
    __m256i b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0 =
        _mm256_shuffle_epi8(
            gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
            _mm256_set_epi8(13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3, 0, /**/ 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3, 0));

    __m256i g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5 =
        _mm256_permute2x128_si256(
            b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
            b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
            1);

    __m256i g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 =
        _mm256_blendv_epi8(
            g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5,
            b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
            _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128));

    // upper lane is not relevant
    __m256i rF_rE_rD_rC_rB_gF_gE_gD_gC_gB_bF_bE_bD_bC_bB_bA =
        _mm256_shuffle_epi8(
            bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
            _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3, 0));

    *rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 = _mm256_blendv_epi8(
        g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
        /*                                           */ rF_rE_rD_rC_rB_gF_gE_gD_gC_gB_bF_bE_bD_bC_bB_bA,
        _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /*lane split*/ 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

    // Blue
    __m256i g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0 =
        _mm256_blendv_epi8(
            g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
            b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
            _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /*lane split*/ 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0));

    *bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0 = _mm256_alignr_epi8(
        rF_rE_rD_rC_rB_gF_gE_gD_gC_gB_bF_bE_bD_bC_bB_bA,
        g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
        6);

    // Green
    __m256i xx_xx_xx_xx_xx_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 =
        _mm256_alignr_epi8(
            g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5,
            b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
            11);

    __m256i gF_gE_gD_gC_gB_rF_rE_rD_rC_rB_bF_bE_bD_bC_bB_bA =
        _mm256_shuffle_epi8(
            bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
            _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2, 13, 10, 7, 4, 1, 15, 12, 9, 6, 3, 0));

    *gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 = _mm256_blendv_epi8(
        xx_xx_xx_xx_xx_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
        gF_gE_gD_gC_gB_rF_rE_rD_rC_rB_bF_bE_bD_bC_bB_bA,
        _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /*lane split*/ 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

    // Move high part of low lane to low part of high lane
    *rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 = _mm256_permute4x64_epi64(*rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0, _MM_SHUFFLE(0, 1, 0, 0));
    *gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 = _mm256_permute4x64_epi64(*gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0, _MM_SHUFFLE(0, 1, 0, 0));
    *bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0 = _mm256_permute4x64_epi64(*bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0, _MM_SHUFFLE(0, 1, 0, 0));

    //Unpack uint8 elements to uint16 elements.
    const __m256i zero_vec = _mm256_setzero_si256();
    *rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 = _mm256_unpacklo_epi8(*rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0, zero_vec);
    *gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 = _mm256_unpacklo_epi8(*gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0, zero_vec);
    *bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0 = _mm256_unpacklo_epi8(*bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0, zero_vec);
}

//Calculate 8 Grayscale elements from 8 RGB elements.
//Y = 0.2989*R + 0.5870*G + 0.1140*B
//Conversion model used by MATLAB https://www.mathworks.com/help/matlab/ref/rgb2gray.html
static __attribute__((always_inline)) inline __m256i Rgb2Yx8(__m256i rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
                                                             __m256i gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
                                                             __m256i bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0)
{
    //Each coefficient is expanded by 2^15, and rounded to int16 (add 0.5 for rounding).
    const __m256i r_coef = _mm256_set1_epi16((short)(0.2126 * 32768.0 + 0.5)); //8 coefficients - R scale factor.
    const __m256i g_coef = _mm256_set1_epi16((short)(0.7152 * 32768.0 + 0.5)); //8 coefficients - G scale factor.
    const __m256i b_coef = _mm256_set1_epi16((short)(0.0722 * 32768.0 + 0.5)); //8 coefficients - B scale factor.

    //Multiply input elements by 64 for improved accuracy.
    rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 = _mm256_slli_epi16(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0, 6);
    gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 = _mm256_slli_epi16(gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0, 6);
    bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0 = _mm256_slli_epi16(bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0, 6);

    //Use the special intrinsic _mm256_mulhrs_epi16 that calculates round(r*r_coef/2^15).
    //Calculate Y = 0.2989*R + 0.5870*G + 0.1140*B (use fixed point computations)
    __m256i y7_y6_y5_y4_y3_y2_y1_y0 = _mm256_add_epi16(_mm256_add_epi16(
                                                           _mm256_mulhrs_epi16(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0, r_coef),
                                                           _mm256_mulhrs_epi16(gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0, g_coef)),
                                                       _mm256_mulhrs_epi16(bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0, b_coef));

    //Divide result by 64.
    y7_y6_y5_y4_y3_y2_y1_y0 = _mm256_srli_epi16(y7_y6_y5_y4_y3_y2_y1_y0, 6);

    return y7_y6_y5_y4_y3_y2_y1_y0;
}

void convert(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    // double accumulator, each doing 16, double of what sse can do (8)
    int floats_per_operation = 16;

    int size = width * height;
    int pixel_per_thread_unaligned = size / threads;
    // Each FMA instruction can calculate 8 pixels at once, so we need a worksize that is a multiple of it.
    // Leftover will need to be handled seperatly without FMA by the last thread.
    int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;

// #pragma omp parallel for
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

        __m256i rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0;
        __m256i gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0;
        __m256i bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0;

        for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
        {
            __m256i r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0 = _mm256_loadu_si256((__m256i *)&img[i * channels]);    //Unaligned load of 32 uint8 elements
            __m256i b7_g7_r7_b6_g6_r6_b5_g5 = _mm256_loadu_si256((__m256i *)&img[i * channels + floats_per_operation * 2]); //Unaligned load of (only) 8 uint8 elements (lower half of XMM register).

            //Separate RGB, and put together R elements, G elements and B elements (together in same XMM register).
            //Result is also unpacked from uint8 to uint16 elements.
            GatherRGBx8(r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
                        b7_g7_r7_b6_g6_r6_b5_g5,
                        &rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
                        &gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
                        &bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0);

            //Calculate 8 Y elements.
            __m256i y7_y6_y5_y4_y3_y2_y1_y0 = Rgb2Yx8(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
                                                      gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
                                                      bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0);

            __m256i j7_j6_j5_j4_j3_j2_j1_j0 = _mm256_packus_epi16(y7_y6_y5_y4_y3_y2_y1_y0, y7_y6_y5_y4_y3_y2_y1_y0);

            j7_j6_j5_j4_j3_j2_j1_j0 = _mm256_permute4x64_epi64(j7_j6_j5_j4_j3_j2_j1_j0, _MM_SHUFFLE(0, 1, 2, 0));

            /*  r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0 = _mm256_loadu_si256((__m256i *)&img[i * channels + 48]); //Unaligned load of 32 uint8 elements
            b7_g7_r7_b6_g6_r6_b5_g5 = _mm256_loadu_si256((__m256i *)&img[i * channels + 80]);                         //Unaligned load of (only) 8 uint8 elements (lower half of XMM register).

            __m256i r7_r6_r5_r4_r3_r2_r1_r02;
            __m256i g7_g6_g5_g4_g3_g2_g1_g02;
            __m256i b7_b6_b5_b4_b3_b2_b1_b02;

            //Separate RGB, and put together R elements, G elements and B elements (together in same XMM register).
            //Result is also unpacked from uint8 to uint16 elements.
            GatherRGBx8(r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
                        b7_g7_r7_b6_g6_r6_b5_g5,
                        &r7_r6_r5_r4_r3_r2_r1_r02,
                        &g7_g6_g5_g4_g3_g2_g1_g02,
                        &b7_b6_b5_b4_b3_b2_b1_b02);

            //Calculate 8 Y elements.
            __m256i y7_y6_y5_y4_y3_y2_y1_y02 = Rgb2Yx8(r7_r6_r5_r4_r3_r2_r1_r02,
                                                       g7_g6_g5_g4_g3_g2_g1_g02,
                                                       b7_b6_b5_b4_b3_b2_b1_b02);
 */
            //Pack uint16 elements to 16 uint8 elements. Only lower lane relevant

            //Store 8 elements of Y in row Y0, and 8 elements of Y in row Y1.
            _mm256_storeu_si256((__m256i *)&result[i], j7_j6_j5_j4_j3_j2_j1_j0);
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
