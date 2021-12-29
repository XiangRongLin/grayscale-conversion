# Benchmarks
## Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz
|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|openmp_baseline|7680x4320|4|406|0.41|
|openmp_baseline|7680x4320|6|441|0.38|
|openmp_baseline|7680x4320|8|492|0.34|
|openmp_baseline|7680x4320|10|468|0.36|
|openmp_baseline|7680x4320|16|433|0.39|

Best performance with 8 threads which makes sense because it is a 4 core/8 logical processor CPU.

|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|openmp_baseline|15360x8640|6|291|2.28|
|openmp_baseline|15360x8640|8|317|2.09|
|openmp_baseline|15360x8640|10|302|2.21|

With a bigger image the best performance is still with 8 threads, but roughly 35% worse.

---

We can optimize the memory access by having each thread only calculate the gray value for a consecutive area.
|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|convert_openmp_memory|7680x4320|8|3443|0.0633|
|convert_openmp_memory|15360x8640|8|3713|0.2478|

This gives us a 650% improvement compared to before for the small image and bringing the big image to the same performance

---

|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|convert_openmp_memory_simd_fma|7680x4320|8|1710|0.0975|
|convert_openmp_memory_simd_fma|15360x8640|8|1790|0.3711|

While keeping the previous adjustments as is but using fused multiply add (FMA) gives a 35% regression in performance.
This could be due to the overhead of the preparation for FMA.
- The single array with the rgb values was split into 3 arrays each only with one of the 3 values.
- Each iteration needs to access values from all 3 arrays. Depending on cache size not all values can be held in cache and need to be reloaded on the next iteration making the memory access continious useless.
- The pixel is saved as 8-bit `unsigned char`, which needs to be converted to a 32-bit `float` for the calculation and then back again to 8-bit `unsigned char` for writing the image.

---

|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|convert_openmp_memory_simd_fma2|7680x4320|8|2204|0.0754|
|convert_openmp_memory_simd_fma2|15360x8640|8|2353|0.2821|

This can be improved by 30% by not splitting up the img into 3 array beforehand, but doing it at the place where it's needed.

---
TODO updated to gcc 11, redo previous benchmarks

With the next change we can calculate 8 numbers at once instead of only 4 giving us a xx% improvement. But it is still worse than just using memory alignment without SIMD.
|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|convert_openmp_memory_simd_fma_256_bit|7680x4320|8|2987|0.0556|
|convert_openmp_memory_simd_fma_256_bit|15360x8640|8|3209|0.2068|

----

TODO using `-O3` now redo benchmarks

Doing the benchmarks with the existing images shows little improvment between using SSE (128 bit vectors) and AVX (256 bit vectors).
But now adding an even bigger image of the size 27000x6000 shows a 100% difference between the 2.

|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|memory|27000x6000|8|9833|0.0827|
|memory_simd_sse|27000x6000|8|6817|0.1199|
|memory_simd_avx|27000x6000|8|12813|0.0647|

---

final after fixing and improving avx

|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|memory_simd_avx|27000x6000|8|16294|0.0501|
