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

__global__ void computeClassVariances(){
// to compute interclass variance 
}
__global__ void findOtsuThresholding(){
// to find the thresh hold for binarize image 
}
__global__ void binarizeImage(){
// applying threshold to get the final binaty image
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

