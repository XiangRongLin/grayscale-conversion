#include <smmintrin.h>
#include <stdio.h>



int main()
{
    unsigned char out[8];
float ins[4] = {10.4, 10.6, 120, 100000};
    __m128 x = _mm_load_ps(ins);       // Load the floats
    __m128i y = _mm_cvtps_epi32(x);    // Convert them to 32-bit ints
    y = _mm_packus_epi32(y, y);        // Pack down to 16 bits
    y = _mm_packus_epi16(y, y);        // Pack down to 8 bits
    *(int*)out = _mm_cvtsi128_si32(y); // Store the lower 32 bits

    printf("%d\n", out[0]);
    printf("%d\n", out[1]);
    printf("%d\n", out[2]);
    printf("%d\n", out[3]);
    return 0;
}