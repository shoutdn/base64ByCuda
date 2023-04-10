
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <fstream>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t decodeWithCuda(const char*src, char*dst, const unsigned int src_size, const unsigned int dst_size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";




//解码base64
//src ,dst都在显存
//sum是 src长度 除以4（有多少个解码组/线程）
__global__ void decodeKernel(char* src, char* dst,int sum)
{
    int i = threadIdx.x;
    char* read = src + 4 * i;
    char* write = dst + 3 * i;

    char read_buffer[4]{ 0 };
    read_buffer[0] = *read;
    read_buffer[1] = *(read + 1);
    read_buffer[2] = *(read + 2);
    read_buffer[3] = *(read + 3);


    if (i != sum) {
        unsigned char_array_4[4]{ 0 };
        for (i = 0; i < 4; +i)
        {
            char c = read_buffer[i];
            char result = NULL;
            if (isalnum(c)) {
                if (isalpha(c)) {

                }
                else {

                }
            }
            else if (c == '+') {

            }
            else if (c == '/') {

            }
            char_array_4[i] = base64_chars.find(read_buffer[i]);
        }
        *(write) = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        *(write+1) = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        *(write+2) = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
    }
    else if(i==sum) {
        unsigned char_array_4[4]{ 0 };
        for (i = 0; i < 4; +i)
        {
            if (read_buffer[i] == '=') {
                char_array_4[i] = base64_chars.find((char)0);
            }
            else {
                char_array_4[i] = base64_chars.find(read_buffer[i]);
            }
        }


        *(write) = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        *(write + 1) = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        *(write + 2) = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
    }

    cudaDeviceSynchronize();

}


int main()
{
    std::string file("./base64.data");
    std::ifstream input(file, std::ios::binary|std::ios::ate);
    unsigned src_size = input.tellg();
    input.seekg(std::ios::beg);

    char* base64 = new char[src_size];
    input.read(base64, src_size);
    input.close();


    int dst_size = src_size / 4 * 3;



    //解码后内存
    char* host_dst = nullptr;
    cudaError_t cudaStatus = cudaHostAlloc((void**)&host_dst, dst_size * sizeof(int), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }




    cudaStatus = decodeWithCuda(base64, host_dst, src_size, dst_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }




    std::ofstream output("output.jpeg", std::ios::binary);
    output.write(host_dst, dst_size * sizeof(int));
    output.close();


    free(base64);
    cudaFreeHost(host_dst);

   

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



cudaError_t decodeWithCuda(const char*& src, char*& dst, const unsigned int src_size,const unsigned int dst_size)
{
    char* dev_src = 0;
    char* dev_dst = 0;
    cudaError_t cudaStatus;

   


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_src, src_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }




    cudaStatus = cudaMalloc((void**)&dev_dst, dst_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }





    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_src, src, src_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }


    int decodeThread = dst_size / 3;
    // Launch a kernel on the GPU with one th   read for each element.
    decodeKernel << <1, decodeThread >> > (dev_src, dev_dst, decodeThread);


    cudaMemcpy(dst, dev_dst, dst_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_src);
    cudaFree(dev_dst);

    return cudaStatus;
}




// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
