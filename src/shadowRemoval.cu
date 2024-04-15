/*
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/

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
__global__ void computeCumulativeSum(unsigned int* probHistogram, unsigned int* cumulativeHistogram, int numBins) {
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
__global__ void computeClassVariances(){
// to compute interclass variance 
}
__global__ void findOtsuThresholding(float* cumSum0, float* cumSum1, int* threshold, int numBins) {
    extern __shared__ float smem[];

    int t = threadIdx.x;
    float* smem0 = smem;
    float* smem1 = smem + numBins;
    float* smemValue = smem + 2 * numBins;
    int* smemIndex = (int*)(smem + 3 * numBins);

    if (t < numBins) {
        smem0[t] = cumSum0[t];
        smem1[t] = cumSum1[t];
        smemValue[t] = 0.0f; // Initialize to zero
        smemIndex[t] = t; // Initialize indices
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
        binaryImage[t] = (imageData[t] >= threshold) ? 255 : 0;
        t += s;
    }
}
__global__ void binarizeImage(){
// applying threshold to get the final binaty image
}
__global__ void calculateOtsuThreshold(unsigned char* histogram, int imageSize, unsigned char* threshold) {
    /*
    __shared__ float cache[256];

    int index = threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }

    float sumB = 0.0f;
    float wB = 0.0f;
    float wF = 0.0f;
    float varMax = 0.0f;

    if (index < 256) {
        for (int t = 0; t < 256; t++) {
            wB += histogram[t];
            if (wB == 0) continue;
            wF = totalPixels - wB;
            if (wF == 0) break;
            sumB += t * histogram[t];
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;
            float varBetween = wB * wF * (mB - mF) * (mB - mF);
            if (varBetween > varMax) {
                varMax = varBetween;
                threshold[0] = t;
            }
        }
    }
    */
}


__global__ void generateMask(const unsigned char *image, unsigned char *mask, const float *cdf, int width, int height, unsigned char* threshold) {
    /*
    // Generate grayscale mask based on thresholding
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int intensity = image[y * width + x];
        mask[y * width + x] = (cdf[intensity] > threshold) ? 255 : 0;
    }
    */
}


/*
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

    // Allocate memory for the histogram
unsigned int* d_histogram;
cudaMalloc((void**)&d_histogram, NUM_BINS * sizeof(unsigned int));
cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));

// Configure kernel launch parameters
int threadsPerBlock = 256;  // This is a typical choice; adjust based on GPU
int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;

// Launch histogram kernel for grayscale image
computeHistogram<<<numBlocks, threadsPerBlock, NUM_BINS * sizeof(unsigned int)>>>(d_grayscaleImage, d_histogram, width * height);

// Configure Erosion Kernel
    // Allocate memory for input image (We will take the gray mask)
        // Do this for sahdow and light mask. 1-mask is the light mask.
    unsigned char* d_erodedMaskShadow;
    unsigned char* d_erodedMaskLight;
    // Allocate input 
        // Cuda host to device copy of the input mask.
    // Allocate output
    cudaMalloc((void**)&d_erodedMaskShadow, grayscaleSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_erodedMaskLight, grayscaleSize * sizeof(unsigned char));
    dim3 erodeBlock(32, 32);
    dim3 erodeGrid(ceil((float)width/erodeBlock.x), ceil((float)height / block.y));

    Erosion<<<erodeGrid, erodeBlock>>>(MASK, d_erodedMaskShadow, width, height, 2)
    //finish erases


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

    // Cleanup
    stbi_image_free(inputImage);
    delete[] colorInvarianceImage;
    delete[] grayscaleImage;
    delete[] UComponentImage;
    delete[] histogram;
    cudaFree(d_inputImage);
    cudaFree(d_colorInvarianceImage);
    cudaFree(d_grayscaleImage);
    cudaFree(d_UComponentImage);
    cudaFree(d_histogram);


    return 0;
}
*/
