#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void ColorTransformation(unsigned char* input, unsigned char* outputColorInvariance, unsigned char* outputGrayscale, unsigned char* outputU, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIndex = y * width + x;
        int rgbIndex = pixelIndex * 3;

        float R = input[rgbIndex];
        float G = input[rgbIndex + 1];
        float B = input[rgbIndex + 2];

        float r_prime = atanf(R / fmaxf(G, B));
        float g_prime = atanf(G / fmaxf(R, B));
        float b_prime = atanf(B / fmaxf(R, G));

        outputColorInvariance[rgbIndex] = static_cast<unsigned char>((r_prime / (M_PI / 2)) * 255);
        outputColorInvariance[rgbIndex + 1] = static_cast<unsigned char>((g_prime / (M_PI / 2)) * 255);
        outputColorInvariance[rgbIndex + 2] = static_cast<unsigned char>((b_prime / (M_PI / 2)) * 255);

        unsigned char grayValue = static_cast<unsigned char>(0.21f * outputColorInvariance[rgbIndex] + 0.71f * outputColorInvariance[rgbIndex + 1] + 0.07f * outputColorInvariance[rgbIndex + 2]);
        outputGrayscale[pixelIndex] = grayValue;

        outputU[pixelIndex] = static_cast<unsigned char>(128 + (-0.147 * R - 0.289 * G + 0.436 * B));
    }
}

#define NUM_BINS 256

__global__ void computeHistogram(unsigned char* imageData, unsigned int* histogram, int numPixels) {
    extern __shared__ unsigned int tempHistogram[];
    int tid = threadIdx.x;
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < NUM_BINS) {
        tempHistogram[tid] = 0;
    }
    __syncthreads();

    // Accumulate histogram in shared memory
    while (pixelIndex < numPixels) {
        atomicAdd(&tempHistogram[imageData[pixelIndex]], 1);
        pixelIndex += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Transfer from shared memory to global histogram
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], tempHistogram[tid]);
    }
}

__global__ void computeCummulativeHistogram(unsigned int* histogram, unsigned int* cumHistogram, int numBins) {
    extern __shared__ unsigned int tempHistogram[];
    int tid = threadIdx.x;
    int n = numBins;

    // Load data into shared memory
    if (tid < n) {
        tempHistogram[tid] = histogram[tid];
    }
    __syncthreads();

    // Up-sweep (Reduction) phase
    for (int offset = 1, d = n >> 1; d > 0; d >>= 1, offset <<= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tempHistogram[bi] += tempHistogram[ai];
        }
    }

    // Set the last element to zero
    if (tid == 0) {
        tempHistogram[n - 1] = 0;
    }

    // Down-sweep phase
    for (int offset = n >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int t = tempHistogram[ai];
            tempHistogram[ai] = tempHistogram[bi];
            tempHistogram[bi] += t;
        }
    }

    __syncthreads();
    // Copy data back to global memory
    if (tid < n) {
        cumHistogram[tid] = tempHistogram[tid];
    }
}


// compute interclass variance
__global__ void comulativeSumHistogram(unsigned int* histogram, unsigned int imageSize, double* zeroOderProbability, double* fisrtOderProbability){
 
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    double zeroOderProbability = 0, zeroOderProbability = 0;
    
    for (int t = 0; t <= id % 256; t++) {
        firstClassProbability += histogram[t];
        firstProbabilitySum += t * histogram[t];
    }

    secondClassProbability = 1 - firstClassProbability;

    
}


// find the thresh hold for binarize image
__global__ void findOtsuThresholding(double* zeroOrderProbability, double* firstOrderProbability, unsigned int* histogram, unsigned char* threshold, int numBins) {
    extern __shared__ double smem[];
    int t = threadIdx.x;
    int n = numBins;

    // Load data into shared memory
    smem[t] = zeroOrderProbability[t];
    smem[n + t] = firstOrderProbability[t];
    __syncthreads();

    // Compute cumulative sum of histograms
    for (int offset = 1; offset < n; offset *= 2) {
        int index = 2 * offset * t;
        if (index < n) {
            smem[n + index + 2 * offset - 1] += smem[n + index + offset - 1];
        }
        __syncthreads();
    }

    // Compute inter-class variances
    double EPSILON = 1e-5;
    double maxVariance = 0.0;
    int maxIndex = 0;
    double denominator;
    for (int i = t; i < n - 1; i += blockDim.x) {
        double numerator = pow((smem[n - 1] * smem[i] - smem[n + i]), 2);
        denominator = smem[i] * (1.0 - smem[i]) + EPSILON;
        double variance = numerator / denominator;
        smem[i] = variance;
        if (variance > maxVariance) {
            maxVariance = variance;
            maxIndex = i;
        }
    }
    __syncthreads();

    // Parallel reduction to find the index of maximum variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && (t + s) < n && smem[t + s] > smem[t]) {
            smem[t] = smem[t + s];
            maxIndex = t + s;
        }
        __syncthreads();
    }

    // Store threshold value in global memory
    if (t == 0) {
        *threshold = maxIndex;
    }
}

// applying threshold to get the final binaty image
__global__ void applyThresholdOnImage(unsigned char* imageData, unsigned char* binaryImage, int imageSize, unsigned char threshold) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    int s = blockDim.x * gridDim.x;

    while (t < imageSize) {
        if (imageData[t] < threshold) {
            binaryImage[t] = 0;
        }
        else {
            binaryImage[t] = 1;
        }
        t += s;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <inputImagePath> <colorInvarianceOutputPath> <grayscaleOutputPath> <UComponentOutputPath>" << std::endl;
        return -1;
    }
    //if (argc < 5) {
    //    std::cerr << "Usage: " << argv[0] << " <inputImagePath> <colorInvarianceOutputPath> <grayscaleOutputPath> <UComponentOutputPath>" << std::endl;
    //    return -1;
    //}

    const char* inputImagePath = argv[1];
    const char* colorInvarianceOutputPath = argv[2];
    const char* grayscaleOutputPath = argv[3];
    const char* UComponentOutputPath = argv[4];

    int width, height, channels;
    unsigned char* inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);

    if (inputImage == NULL) {
        std::cerr << "Error loading image: " << inputImagePath << std::endl;
        return -1;
    }

    if (channels < 3) {
        std::cerr << "Error: Image must have at least 3 channels (RGB)" << std::endl;
        stbi_image_free(inputImage);
        return -1;
    }

    int imageSize = width * height * channels;
    int grayscaleSize = width * height;

    unsigned char *d_inputImage, *d_colorInvarianceImage, *d_grayscaleImage, *d_UComponentImage;
    cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_colorInvarianceImage, imageSize * sizeof(unsigned char)); // 3 channels
    cudaMalloc((void**)&d_grayscaleImage, grayscaleSize * sizeof(unsigned char)); // Single channel
    cudaMalloc((void**)&d_UComponentImage, grayscaleSize * sizeof(unsigned char)); // U component

    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the combined kernel
    ColorTransformation<<<gridDim, blockDim>>>(d_inputImage, d_colorInvarianceImage, d_grayscaleImage, d_UComponentImage, width, height);
    // Allocate memory for output images
    unsigned char* colorInvarianceImage = new unsigned char[imageSize];
    unsigned char* grayscaleImage = new unsigned char[grayscaleSize];
    unsigned char* UComponentImage = new unsigned char[grayscaleSize];

    // Copy the converted images back to host
    cudaMemcpy(colorInvarianceImage, d_colorInvarianceImage, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(grayscaleImage, d_grayscaleImage, grayscaleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(UComponentImage, d_UComponentImage, grayscaleSize, cudaMemcpyDeviceToHost);

    // Save the output images
    stbi_write_jpg(colorInvarianceOutputPath, width, height, 3, colorInvarianceImage, 100);
    stbi_write_jpg(grayscaleOutputPath, width, height, 1, grayscaleImage, 100);
    stbi_write_jpg(UComponentOutputPath, width, height, 1, UComponentImage, 100);

    // Allocate memory for the histogram
unsigned int* d_histogram;
cudaMalloc((void**)&d_histogram, NUM_BINS * sizeof(unsigned int));
cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));

// Configure kernel launch parameters
int threadsPerBlock = 256;  // This is a typical choice; adjust based on GPU
int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;

// Launch histogram kernel for grayscale image
computeHistogram<<<numBlocks, threadsPerBlock, NUM_BINS * sizeof(unsigned int)>>>(d_grayscaleImage, d_histogram, width * height);

// Copy histogram back to host
unsigned int* histogram = new unsigned int[NUM_BINS];
cudaMemcpy(histogram, d_histogram, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //Printing histogram values
    printf("Histogram values:\n");
    for (int i = 0; i < NUM_BINS; i++) {
        printf("%d: %u, ", i, histogram[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");


    // Allocate memory for the cumulative histogram
    unsigned int* d_cumulativeHistogram;
    cudaMalloc((void**)&d_cumulativeHistogram, NUM_BINS * sizeof(unsigned int));

    // Launch cumulative histogram kernel
    // Assuming the histogram size (256) fits within a single block's maximum threads
    computeCummulativeHistogram << <1, NUM_BINS, NUM_BINS * sizeof(unsigned int) >> > (d_histogram, d_cumulativeHistogram, NUM_BINS);

    // Optionally, retrieve the cumulative histogram to host for verification or further processing
    unsigned int* cumulativeHistogram = new unsigned int[NUM_BINS];
    cudaMemcpy(cumulativeHistogram, d_cumulativeHistogram, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    

    // Cleanup
    stbi_image_free(inputImage);
    delete[] colorInvarianceImage;
    delete[] grayscaleImage;
    delete[] UComponentImage;
    delete[] histogram;
    delete[] cumulativeHistogram;

    cudaFree(d_inputImage);
    cudaFree(d_colorInvarianceImage);
    cudaFree(d_grayscaleImage);
    cudaFree(d_UComponentImage);
    cudaFree(d_histogram);
    cudaFree(d_cumulativeHistogram);
    cudaFree(d_histogram);
    return 0;
}

