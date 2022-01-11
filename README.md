Report by [Dennis Weggenmann](https://github.com/DennisWeggenmann) and [Xiang Rong Lin](https://github.com/XiangRongLin) for the lecture "High Performance Computing" at the "Hochschule für Technik Stuttgart"

# Motivation

# GPU

## Problem

## Solution attempt

## Implementation

## Review

## Conclusion

# CPU
## Problem
- naive solution is single threaded
- processors can calculate 128/256 bit at once, but only part of it is used in a single iteration
- data is in rgbrgbrgbrgb format, but rrrrggggbbbb is needed
- memory is not aligned

## Solution attempt
### Multithreaded
The easiest place to optimize it, is utilizing all cores of a CPU and thus convert it to a multithreaded application.
This is done with OpenMP by adding the pragma `#pragma omp parallel for collapse(2)` (see [openmp_baseline.c](cpu/algorithms/openmp_baseline.c))).
`omp parallel for` parallelizes the `for` loop with `collapse(2)` collapsing both loops and thus parallelizing both.
This gives a more than 6 times performance boost.

In the next step the memory access can optimized.
Currently each thread calculates the grey value for a random pixel, depending on how it is scheduled by openMP.
For this it needs to load 3 unsigned char, so 24 bytes from memory.
But a CPU preloads more data into the cache anticipating that it will be needed.
This behavior can be used to by having each thread operating on a continuous section, thus using the data that is already in the CPU Cache (see [memory.c](cpu/algorithms/memory.c)).

### SIMD FMA
All references to intrinsic functions can be looked up here: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

The next place to optimize is utilizing the full register of the CPU by using Single Instruction Multiple Data (SIMD).
For example we could add 16 8-bit integers at once in a 128 bit register instead of only a single one, thus theoretically creating a 16 time speedup.
Additionally one can use a dedicated arithmetic logic unit that multiplies two numbers and add it to an accumulator, knows as MAC-unit (multiplier-accumulator).
This takes the form of "fused multiply add" (FMA), which additionally only rounds at the end, thus combining two operations into one.

The data being in the form of rgbrgbrgbrgb appears for the first time.
For FMA a whole register need to be filled with only red, green or blue values, meaning we want the data in rrrrggggbbbb format.
This problem is ignored for now, by just setting the register with the appropriate values, which comes with its own performance problems, because with the data being spread out like this, multiple reads may be necessary.
```C
r_vector = _mm_set_ps(img[(i * channels)], img[(i + 1) * channels], img[(i + 2) * channels], img[(i + 3) * channels]);
g_vector = _mm_set_ps(img[(i * channels) + 1], img[(i + 1) * channels + 1], img[(i + 2) * channels + 1], img[(i + 3) * channels + 1]);
b_vector = _mm_set_ps(img[(i * channels) + 2], img[(i + 1) * channels + 2], img[(i + 2) * channels + 2], img[(i + 3) * channels + 2]);
```

With the data in the correct format the multiplication is very simple
```C
gray_vector = _mm_setzero_ps();
gray_vector = _mm_fmadd_ps(r_vector, r_factor, gray_vector);
gray_vector = _mm_fmadd_ps(g_vector, g_factor, gray_vector);
gray_vector = _mm_fmadd_ps(b_vector, b_factor, gray_vector);
```

Full code see [memory_simd_fma.c](cpu/algorithms/memory_simd_fma.c)

A problem with FMA is, that the basic FMA instruction set only supports working with 32-bit and 64-bit floating point numbers.
This means that with a 128-bit register a maximum of 4 pixel can be calculated at once.

### SIMD SSE


## Benchmarks
With 
- AMD Ryzen 5 3600 6-Core Processor 
- gcc 11
- compiled `gcc -fopenmp grayscale.c -lm -march=native -O`
- 20 runs each
- 27000x6000 pixel image https://photojournal.jpl.nasa.gov/catalog/?IDNumber=PIA03239

### Baseline
|time in s|megapixel per s|
|---|---|
|2.739997|56.3852|

### openmp baseline

|thread number|time in s|megapixel per s|
|---|---|---|
|12|0.411790|375.1795|
|32|0.414154|373.0381|
|64|0.414555|372.6776|
|128|0.430195|359.1287|

### memory

|thread number|time in s|megapixel per s|
|---|---|---|
|32|0.032053|4820.0608|
|64|0.031559|4895.3717|
|128|0.030711|5030.6157|
|256|0.032755|4716.6270|

### FMA

|thread number|time in s|megapixel per s|
|---|---|---|
|12|0.035387|4365.8509|
|32|0.035855|4308.9137|
|64|0.035242|4383.8510|
|128|0.034176|4520.5836|
|256|0.035138|4396.8387|

### SSE

|thread number|time in s|megapixel per s|
|---|---|---|
|12|0.030364|5088.0302|
|32|0.030036|5143.7032|
|64|0.030248|5107.6352|
|128|0.030838|5009.9306|
|256|0.032062|4818.7077|

### AVX

|thread number|time in s|megapixel per s|
|---|---|---|
|12|0.030029|5144.9279|
|32|0.029775|5188.7483|
|64|0.030188|5117.7360|
|128|0.030685|5034.9111|
|256|0.032302|4782.7716|

### CPU comparison
|CPU|algorithm|thread number|time in s|megapixel per s|
|---|---|---|---|---|
|AMD Ryzen 5 3600 (6 Core)|simd_sse|32|0.030036|5143.7032|
|AMD Ryzen 5 3600 (6 Core)|simd_avx|32|0.029775|5188.7483|
|Intel Core i9-9880H (8 Core)|simd_sse|64|0.027484|5621.3304|
|Intel Core i9-9880H (8 Core)|simd_avx|128|0.027279|5663.5644|

## Review
### Memory Bottleneck
27000*6000 = 486.000.000 (pixel)
486.000.000 * 3 (bytes) = 463,49 (mb)
on average 30ms for 463,49mb of image => 15459mb/s data transfer => 15gb/s data transfer
memory bandwidth of CPU is 48gb/s in dual channel mode. https://en.wikichip.org/wiki/amd/ryzen_5/3600
But with the relativly small 
=> probably no

## Conclusion
