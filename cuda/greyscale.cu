#include <math.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../baseline/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../baseline/stb_image_write.h"
#include <time.h>


#include <algorithm>

#define CHANNELS 3

//this function runs on the device(gpu)
__global__
void ConvertToGrey(uchar3  *input, unsigned char *output, int rows, int columns)
{
    // http://algogroup.unimore.it/people/marko/courses/programmazione_parallela/PP1718/Lecture-3_Parallelism_model.pdf
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (index_x < columns && index_y < rows)
	{
		int output_offset = index_y * columns + index_x;
       uchar3  rgb = input[output_offset];	
	    output[output_offset] = rgb.x * 0.299f +rgb.y* 0.587f +rgb.z * 0.114f ;
    }
}

//runs on the host
int main (){
    int rows,columns,channels,pixel_size;
    unsigned char* Image = stbi_load("../images/15360x8640.jpg", &columns, &rows, &channels, 0);
    uchar3 *d_rgb_image;
    unsigned char *h_grey_image, *d_grey_image;

    pixel_size = columns * rows ;
/*
     printf("Image size: %d x %d\n", columns, rows);
        printf("Pixel size: %d\n", pixel_size);
        printf("Channels: %d\n", channels);
*/
    h_grey_image = (unsigned char *)malloc(sizeof(unsigned char*)* pixel_size);

    //Allocate device memory for the image
    cudaMalloc(&d_rgb_image, sizeof(uchar4) * pixel_size * CHANNELS);
    //allocate device memory for the grey image
	cudaMalloc(&d_grey_image, sizeof(unsigned char) * pixel_size);
    //sets device memory to a value.
	cudaMemset(d_grey_image, 0, sizeof(unsigned char) * pixel_size);

    int thread = 16;
    const dim3 Block(thread, thread);
    const dim3 Grid((columns + Block.x - 1) / Block.x, (rows + Block.y - 1) / Block.y);

    clock_t start, end;

    // measure the time taken to convert the image to grey with copy to device and back to host
    start = clock();
    cudaMemcpy(d_rgb_image, Image, sizeof(unsigned char) * pixel_size * CHANNELS, cudaMemcpyHostToDevice);
    
    //calls the kernel function
    ConvertToGrey<<<Grid, Block>>>(d_rgb_image, d_grey_image, rows, columns);
        
    // i dont think i need this but i will leave it here..
    // cudaDeviceSynchronize();
    end = clock();
    double time =(double)(end-start)/CLOCKS_PER_SEC;
    printf("gpu execution and copy time is %.30lf\n", time);

    // Copy the data back to the host
    cudaMemcpy(h_grey_image, d_grey_image, sizeof(unsigned char) * pixel_size, cudaMemcpyDeviceToHost);
    
    // stbi_write_jpg("../images/grey.jpg", columns, rows, 1, h_grey_image, 100);

    // free the allocated memory on the host and the device
    free(h_grey_image);
    cudaFree(d_rgb_image);
    cudaFree(d_grey_image);

    return 0;
	
}