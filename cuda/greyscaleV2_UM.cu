#include <math.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../baseline/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../baseline/stb_image_write.h"
#include <time.h>

__global__
void setPixelToGrayscale(unsigned char *image, int rows, int columns)
{
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;
	if (index_x < columns && index_y < rows){

	int index = index_y * columns + index_x;
	int rgb_offset = index * 3;
  		unsigned char r = image[rgb_offset + 0];
	    unsigned char g = image[rgb_offset + 1];
	    unsigned char b = image[rgb_offset + 2];
    image[index] = .299f*r + .587f*g + .114f*b;
}
}

int main()
{
	int width, height, channels;
    unsigned char* image = stbi_load("../images/PIA18164.jpg", &width, &height, &channels, 0);
    unsigned char* image_d;
    int N = width * height; 
    size_t size = N * sizeof(unsigned char) *3;
	 int thread = 32;
    const dim3 Block(thread, thread);
    const dim3 Grid((width + Block.x - 1) / Block.x, (height + Block.y - 1) / Block.y);
	  cudaMallocManaged(&image_d, size);
    setPixelToGrayscale<<<Grid,Block>>>(image_d,height,width);
    cudaDeviceSynchronize();
	// stbi_write_png("../images/.grey.png", width, height, 1,image, width );
//	 stbi_write_jpg("../images/grey.jpg", width, height, 3, image, 100);
    cudaFree(image_d);
}