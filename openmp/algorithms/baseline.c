void convert_baseline(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
#pragma omp parallel for collapse(2)
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            result[y * width + x] =
                0.2126 * img[(y * width + x) * channels]        // red
                + 0.7152 * img[(y * width + x) * channels + 1]  // green
                + 0.0722 * img[(y * width + x) * channels + 2]; // blue
        }
    }
}