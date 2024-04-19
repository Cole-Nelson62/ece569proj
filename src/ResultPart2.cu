#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to perform parallel reduction
__global__ void sumArray(double *array, double *result, int N) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N) {
        sdata[tid] = array[i];
    } else {
        sdata[tid] = 0.0;
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1024; // Size of the array
    const int blockSize = 256; // Threads per block
    const int numBlocks = (N + blockSize - 1) / blockSize; // Number of blocks needed

    // Allocate memory on the host for the array
    double *h_array = new double[N];

    // Initialize array values (for example)
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    // Allocate memory on the device for the array and result
    double *d_array, *d_result;
    cudaMalloc((void **)&d_array, N * sizeof(double));
    cudaMalloc((void **)&d_result, numBlocks * sizeof(double));

    // Copy array data from host to device
    cudaMemcpy(d_array, h_array, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    sumArray<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_array, d_result, N);

    // Copy the result back to the host
    double *h_result = new double[numBlocks];
    cudaMemcpy(h_result, d_result, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Perform final reduction on the CPU (summing up block results)
    double finalResult = 0.0;
    for (int i = 0; i < numBlocks; ++i) {
        finalResult += h_result[i];
    }

    std::cout << "Sum of array elements: " << finalResult << std::endl;

    // Clean up memory
    delete[] h_array;
    delete[] h_result;
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}