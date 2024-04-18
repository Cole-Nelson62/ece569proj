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



#define NUM_BINS 256
__global__ void computeCumulativeSum(unsigned int* histogram, int imageSize, float* cumulativeSum0, float* cumulativeSum1) {
    extern __shared__ float smem[];  // This shared memory is twice the size of NUM_BINS to store both sums
    float* smem0 = smem;
    float* smem1 = smem + NUM_BINS;

    int t = threadIdx.x;
    int n = NUM_BINS / 2;  // Use half the number of bins per block

    // Load data into shared memory with offsets to avoid bank conflicts
    if (t < n) {
        int offset1 = (t >> 4);
        int offset2 = ((t + n) >> 4);
        smem0[t + offset1] = (float)histogram[t] / imageSize;
        smem1[t + offset1] = (float)histogram[t] * t / imageSize;
        smem0[t + n + offset2] = (float)histogram[t + n] / imageSize;
        smem1[t + n + offset2] = (float)histogram[t + n] * (t + n) / imageSize;
    }
    __syncthreads();

    // Up-sweep (reduction) phase
    for (int d = 1; d < NUM_BINS; d *= 2) {
        int stride = d << 1;
        if ((t % stride) == 0) {
            int index1 = t + d - 1 + (t + d - 1) / 16;
            int index2 = t + stride - 1 + (t + stride - 1) / 16;
            smem0[index2] += smem0[index1];
            smem1[index2] += smem1[index1];
        }
        __syncthreads();
    }

    // Clear the last element
    if (t == 0) {
        int last = NUM_BINS - 1 + (NUM_BINS - 1) / 16;
        smem0[last] = 0;
        smem1[last] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int d = NUM_BINS / 2; d >= 1; d /= 2) {
        int stride = d << 1;
        if ((t % stride) == 0) {
            int index1 = t + d - 1 + (t + d - 1) / 16;
            int index2 = t + stride - 1 + (t + stride - 1) / 16;
            float temp0 = smem0[index1];
            float temp1 = smem1[index1];
            smem0[index1] = smem0[index2];
            smem1[index1] = smem1[index2];
            smem0[index2] += temp0;
            smem1[index2] += temp1;
        }
        __syncthreads();
    }

    // Write back to global memory
    if (t < n) {
        int outputIndex1 = t + (t >> 4);
        int outputIndex2 = t + n + ((t + n) >> 4);
        cumulativeSum0[t] = smem0[outputIndex1];
        cumulativeSum1[t] = smem1[outputIndex1];
        cumulativeSum0[t + n] = smem0[outputIndex2];
        cumulativeSum1[t + n] = smem1[outputIndex2];
    }
}

#define NUM_BINS 256
#define EPSILON 1e-10

__global__ void findOtsuThresholding(float *cumulativeSum0, float *cumulativeSum1, int *optimalThreshold) {
    extern __shared__ float smem[];
    float *smemCumSum0 = smem;                    // First half for 0th order cumulative sum
    float *smemCumSum1 = smem + NUM_BINS;         // Second half for 1st order cumulative sum
    float *smemValue = smem + 2 * NUM_BINS;       // Values of inter-class variances
    int *smemIndex = (int *)(smem + 3 * NUM_BINS); // Indices for tracking maximum variance location

    int t = threadIdx.x;
    int n = NUM_BINS;

    // Load data into shared memory
    if (t < n) {
        smemCumSum0[t] = cumulativeSum0[t];
        smemCumSum1[t] = cumulativeSum1[t];
        smemValue[t] = 0;  // Initialize variance storage
        smemIndex[t] = t;  // Initialize indices
    }
    __syncthreads();

    // Compute inter-class variances
    if (t < n) {
        float w0 = smemCumSum0[t];
        float w1 = 1 - w0;
        float mu0 = (w0 > EPSILON) ? smemCumSum1[t] / w0 : 0;
        float mu1 = (w1 > EPSILON) ? (smemCumSum1[n - 1] - smemCumSum1[t]) / w1 : 0;
        float numerator = (mu0 - mu1) * (mu0 - mu1);
        float denominator = w0 * w1 + EPSILON;
        smemValue[t] = numerator / denominator;
    }
    __syncthreads();

    // Parallel reduction to find the maximum variance
    for (int s = NUM_BINS / 2; s > 0; s >>= 1) {
        if (t < s) {
            if (smemValue[t + s] > smemValue[t]) {
                smemValue[t] = smemValue[t + s];
                smemIndex[t] = smemIndex[t + s];
            }
        }
        __syncthreads();
    }

    // Copy the index of maximum variance to global memory
    if (t == 0) {
        *optimalThreshold = smemIndex[0];
    }
}

__global__ void applyThreshold(const unsigned char* imageData, unsigned char* binaryImage, int imageSize, int threshold) {
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
