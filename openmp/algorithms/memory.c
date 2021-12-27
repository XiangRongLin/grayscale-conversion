void convert(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
    int pixel_per_thread = (width * height) / threads;
#pragma omp parallel for
    for (int thread = 0; thread < threads; thread++)
    {
        int end;
        if (thread + 1 == threads)
        {
            end = width * height;
        }
        else
        {
            end = pixel_per_thread * (thread + 1);
        }

        for (int i = pixel_per_thread * thread; i < end; i++)
        {
            result[i] =
                0.2126 * img[(i * channels)]    // red
                + 0.7152 * img[(i * channels) + 1]  // green
                + 0.0722 * img[(i * channels) + 2]; // blue
        }
    }
}