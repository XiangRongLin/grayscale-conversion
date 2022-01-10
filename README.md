

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
### Multi Threaded
First

## Implementation

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
|0.035498|4352.2054|

### Memory

|thread number|time in s|megapixel per s|
|---|---|---|
|2.739997| 56.3852|

### FMA

|thread number|time in s|megapixel per s|
|---|---|---|
12|0.035387|4365.8509|
32|0.035855|4308.9137|
64|0.035242|4383.8510|
128|0.034176|4520.5836|
256|0.035138|4396.8387|

### SSE

thread number|time in s|megapixel per s|
---|---|---|---|
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
|Intel Core i9-9880H (8 Core)|simd_sse|24|0.028947||
|Intel Core i9-9880H (8 Core)|simd_avx|24|0.027712||

## Review

## Conclusion