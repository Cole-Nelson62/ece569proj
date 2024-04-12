#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NUM_BINS 256
#define EPSILON 1e-10
__global__ void computeHistogram(unsigned char *imageData, int imageSize, unsigned int *histogram, int numBins) {
    extern __shared__ unsigned int histSmem[];  // Use unique name for shared memory

    int t = threadIdx.x;
    int p = threadIdx.x + blockIdx.x * blockDim.x;
    int q = blockDim.x * gridDim.x;

    if (t < numBins) {
        histSmem[t] = 0;
    }
    __syncthreads();

    while (p < imageSize) {
        unsigned char value = imageData[p];
        atomicAdd(&histSmem[value], 1);
        p += q;
    }
    __syncthreads();

    if (t < numBins) {
        atomicAdd(&histogram[t], histSmem[t]);
    }
}

__global__ void computeCumulativeSum(unsigned int *probHistogram, unsigned int *cumulativeHistogram, int numBins) {
    extern __shared__ unsigned int cumSmem[];  // Unique name for shared memory

    int t = threadIdx.x;
    int p = t;
    int q = t + numBins / 2;

    if (p < numBins) {
        cumSmem[p] = probHistogram[p];
    }
    if (q < numBins) {
        cumSmem[q] = probHistogram[q];
    }

    __syncthreads();

    int offset = 1;
    for (int d = numBins >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            p = offset * (2 * t + 1) - 1;
            q = offset * (2 * t + 2) - 1;
            cumSmem[q] = cumSmem[q] + cumSmem[p];
        }
        offset *= 2;
    }

    __syncthreads();

    if (t == 0) {
        cumulativeHistogram[numBins - 1] = cumSmem[numBins - 1];
        cumSmem[numBins - 1] = 0;
    }

    __syncthreads();

    for (int d = 1; d < numBins; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            p = offset * (2 * t + 1) - 1;
            q = offset * (2 * t + 2) - 1;
            unsigned int temp = cumSmem[p];
            cumSmem[p] = cumSmem[q];
            cumSmem[q] += temp;
        }
    }

    __syncthreads();

    if (p < numBins) {
        cumulativeHistogram[p] = cumSmem[p];
    }
    if (q < numBins - 1) {
        cumulativeHistogram[q] = cumSmem[q];
    }
}
__global__ void computeInterClassVariances(float* cumSum0, float* cumSum1, int* threshold, int numBins) {
    extern __shared__ float smem[];

    int t = threadIdx.x;
    float* smem0 = smem;
    float* smem1 = smem + numBins;
    float* smemValue = smem + 2 * numBins;
    int* smemIndex = (int*)(smem + 3 * numBins);

    if (t < numBins) {
        smem0[t] = cumSum0[t];
        smem1[t] = cumSum1[t];
        smemIndex[t] = t;
    }
    __syncthreads();

    if (t < numBins) {
        float numerator = pow((smem1[numBins - 1] * smem0[t] - smem1[t]), 2);
        float denominator = smem0[t] * (1 - smem0[t]) + EPSILON;
        smemValue[t] = numerator / denominator;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && t + s < numBins) {
            if (smemValue[t + s] > smemValue[t]) {
                smemValue[t] = smemValue[t + s];
                smemIndex[t] = smemIndex[t + s];
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        *threshold = smemIndex[0];
    }
}

__global__ void applyThreshold(const unsigned char* imageData, unsigned char* binaryImage, int imageSize, unsigned char threshold) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    int s = blockDim.x * gridDim.x;

    while (t < imageSize) {
        binaryImage[t] = (imageData[t] >= threshold) ? 1 : 0;
        t += s;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input image path> <output image path>\n", argv[0]);
        exit(1);
    }

    char *inputImagePath = argv[1];
    char *outputImagePath = argv[2];

    int width, height, channels;
    unsigned char *host_imageData = stbi_load(inputImagePath, &width, &height, &channels, 1);
    if (host_imageData == NULL) {
        fprintf(stderr, "Error in loading the image %s\n", inputImagePath);
        exit(1);
    }
    int imageSize = width * height;

    // Device memory pointers
    unsigned char *dev_imageData, *dev_binaryImage;
    unsigned int *dev_histogram, *dev_cumulativeHistogram;
    float *dev_cumSum0, *dev_cumSum1;
    int *dev_threshold;

    cudaMalloc(&dev_imageData, imageSize * sizeof(unsigned char));
    cudaMalloc(&dev_binaryImage, imageSize * sizeof(unsigned char));
    cudaMalloc(&dev_histogram, NUM_BINS * sizeof(unsigned int));
    cudaMalloc(&dev_cumulativeHistogram, NUM_BINS * sizeof(unsigned int));
    cudaMalloc(&dev_cumSum0, NUM_BINS * sizeof(float));
    cudaMalloc(&dev_cumSum1, NUM_BINS * sizeof(float));
    cudaMalloc(&dev_threshold, sizeof(int));

    cudaMemcpy(dev_imageData, host_imageData, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (imageSize + threadsPerBlock - 1) / threadsPerBlock;

    computeHistogram<<<blocksPerGrid, threadsPerBlock, NUM_BINS * sizeof(unsigned int)>>>(dev_imageData, imageSize, dev_histogram, NUM_BINS);
    computeCumulativeSum<<<1, NUM_BINS / 2, NUM_BINS * sizeof(unsigned int)>>>(dev_histogram, dev_cumulativeHistogram, NUM_BINS);
    computeInterClassVariances<<<1, threadsPerBlock, 4 * NUM_BINS * sizeof(float) + NUM_BINS * sizeof(int)>>>(dev_cumSum0, dev_cumSum1, dev_threshold, NUM_BINS);

    int threshold;
    cudaMemcpy(&threshold, dev_threshold, sizeof(int), cudaMemcpyDeviceToHost);

    applyThreshold<<<blocksPerGrid, threadsPerBlock>>>(dev_imageData, dev_binaryImage, imageSize, threshold);

    unsigned char *binaryImage = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
    cudaMemcpy(binaryImage, dev_binaryImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg(outputImagePath, width, height, 1, binaryImage, 100);

    stbi_image_free(host_imageData);
    free(binaryImage);
    cudaFree(dev_imageData);
    cudaFree(dev_binaryImage);
    cudaFree(dev_histogram);
    cudaFree(dev_cumulativeHistogram);
    cudaFree(dev_cumSum0);
    cudaFree(dev_cumSum1);
    cudaFree(dev_threshold);

    return 0;
}
