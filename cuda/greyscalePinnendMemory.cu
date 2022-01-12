#include <math.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../baseline/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../baseline/stb_image_write.h"
#include <time.h>


#include <algorithm>

//this function runs on the device(gpu)
__global__
void ConvertToGrey(uchar3  *input, unsigned char *output, int rows, int columns)
{
    // http://algogroup.unimore.it/people/marko/courses/programmazione_parallela/PP1718/Lecture-3_Parallelism_model.pdf
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (index_x < columns && index_y < rows)
	{
        //1d coordinate of the image
		int output_offset = index_y * columns + index_x;
        uchar3  rgb = input[output_offset];	
	    output[output_offset] = rgb.x * 0.299f +rgb.y* 0.587f +rgb.z * 0.114f ;
    }
}

//runs on the host
int main (){
    int rows,columns,channels,pixel_size;
    unsigned char* Image = stbi_load("../images/PIA03239.jpg", &columns, &rows, &channels, 0);
    uchar3 *device_rgb;
    unsigned char *host_grey, *device_grey;
    cudaError_t result;
    pixel_size = columns * rows ;

    cudaFree(0);
    result =  cudaHostRegister(Image, pixel_size *3, cudaHostRegisterPortable);	
    //for profiling purposes

    cudaMallocHost(&host_grey, sizeof(unsigned char)* pixel_size);
    //Allocate device memory for the image
   cudaMalloc(&device_rgb, sizeof(uchar4) * pixel_size*3 );
    //allocate device memory for the grey image
	cudaMalloc(&device_grey, sizeof(unsigned char) * pixel_size);
    //sets device memory to a value.
	cudaMemset(device_grey, 0, sizeof(unsigned char) * pixel_size);

    if(result != cudaSuccess) {
    printf("Error: cudaHostRegister returned %s (code %d)\n", cudaGetErrorString(result), result);
    printf("Error in cudaHostRegister: %s.\n", cudaGetErrorString(result));
    return -1;
    }

    int thread = 16;
    const dim3 Block(thread, thread);
    const dim3 Grid((columns + Block.x - 1) / Block.x, (rows + Block.y - 1) / Block.y);

    clock_t start, end;

    // measure the time taken to convert the image to grey with copy to device and back to host
    start = clock();
    cudaMemcpyAsync(device_rgb, Image, sizeof(unsigned char) * pixel_size*3 , cudaMemcpyHostToDevice,0);
    
    //calls the kernel function
    ConvertToGrey<<<Grid, Block>>>(device_rgb, device_grey, rows, columns);
        
    // i dont think i need this but i will leave it here..
    // cudaDeviceSynchronize();
   
    // Copy the data back to the host
    cudaMemcpy(host_grey, device_grey, sizeof(unsigned char) * pixel_size, cudaMemcpyDeviceToHost);

     end = clock();
    double time =(double)(end-start)/CLOCKS_PER_SEC;
    printf("gpu execution and copy time is %.30lf\n", time);
    
    // stbi_write_jpg("../images/grey.jpg", columns, rows, 1, host_grey, 100);

  //   stbi_write_png("../images/.grey.png", columns, rows,1,host_grey, columns );

    // free the allocated memory on the host and the device
    cudaFree(host_grey);
    cudaFree(device_rgb);
    cudaFree(device_grey);

    return 0;
	
}