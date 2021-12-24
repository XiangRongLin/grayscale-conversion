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

We can optimize the memory access by having each thread only calculate the gray value for a consecutive area.
|Name|image|thread number|MFLOPS/s|time in s|
|---|---|---|---|---|
|convert_openmp_memory|7680x4320|8|2628|0.0633|
|convert_openmp_memory|15360x8640|8|2682|0.2478|

This gives us a 650% improvement compared to before for the small image and bringing the big image to the same performance