# Install openMP
It is bundled with you gcc installation.
## Mingw
Use `mingw-get install mingw32-pthreads-w32`

# How to run
- Compile with: `gcc -fopenmp grayscale.c -lm -march=native -O3`
- Download the 27000x6000 image from specified here https://github.com/XiangRongLin/grayscale-conversion/blob/master/cpu/grayscale.c#L54 or rename it to one that is provided
- Run the executable with following arguments for example `10 12 0 1`. Run it without any argument to get information what each number does.

# Common errors
- inlining failed in call to ‘always_inline’ ‘\<function name\>’: target specific option mismatch
    - Your CPU does not support the used SIMD functions.

# Ideas
- Use half precision floats (16-bit ph instead of 32-bit ps). Not supported on my CPU