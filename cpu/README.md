# Install openMP
It is bundled with you gcc installation.
## Mingw
Use `mingw-get install mingw32-pthreads-w32`

# Common errors
- inlining failed in call to ‘always_inline’ ‘\<function name\>’: target specific option mismatch
    - Your CPU does not support the used SIMD functions.

# Ideas
- Use half precision floats (16-bit ph instead of 32-bit ps). Not supported on my CPU