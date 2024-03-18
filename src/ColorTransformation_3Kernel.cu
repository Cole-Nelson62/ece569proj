#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Kernel to convert RGB to Color Invariance
__global__ void colorInvarianceConversion(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = y * width + x;
        int rgbOffset = offset * 3;
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];

        float r_prime = atanf(r / fmaxf(g, b));
        float g_prime = atanf(g / fmaxf(r, b));
        float b_prime = atanf(b / fmaxf(r, g));

        // Scale and convert to 0-255 range
        output[rgbOffset] = static_cast<unsigned char>((r_prime / (M_PI / 2)) * 255);
        output[rgbOffset + 1] = static_cast<unsigned char>((g_prime / (M_PI / 2)) * 255);
        output[rgbOffset + 2] = static_cast<unsigned char>((b_prime / (M_PI / 2)) * 255);
    }
}

// Kernel to convert Color Invariance RGB to Grayscale
__global__ void grayscaleConversion(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int grayOffset = y * width + x;
        int rgbOffset = grayOffset * 3;
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];

        // Directly use transformed color values to compute grayscale
        unsigned char grayValue = static_cast<unsigned char>(0.212f * r + 0.715f * g + 0.072f * b);

        output[grayOffset] = grayValue;
    }
}

// Kernel to compute the U component of YUV
__global__ void computeUComponent(unsigned char* input, unsigned char* outputU, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Index for the RGB values
        float R = input[idx];
        float G = input[idx + 1];
        float B = input[idx + 2];

        // Compute U component
        unsigned char U = static_cast<unsigned char>(128 + (-0.147 * R - 0.289 * G + 0.436 * B));
        outputU[y * width + x] = U;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <inputImagePath> <colorInvarianceOutputPath> <grayscaleOutputPath> <UComponentOutputPath>" << std::endl;
        return -1;
    }

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

    unsigned char *d_inputImage, *d_colorInvarianceImage, *d_grayscaleImage, *d_UComponentImage;
    cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_colorInvarianceImage, imageSize * sizeof(unsigned char)); // 3 channels
    cudaMalloc((void**)&d_grayscaleImage, width * height * sizeof(unsigned char)); // Single channel
    cudaMalloc((void**)&d_UComponentImage, width * height * sizeof(unsigned char)); // U component

    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Convert to color invariance
    colorInvarianceConversion<<<gridDim, blockDim>>>(d_inputImage, d_colorInvarianceImage, width, height);

    // Convert color invariance to grayscale
    grayscaleConversion<<<gridDim, blockDim>>>(d_colorInvarianceImage, d_grayscaleImage, width, height);

    // Compute U component
    computeUComponent<<<gridDim, blockDim>>>(d_inputImage, d_UComponentImage, width, height);

    // Allocate memory for output images
    unsigned char* colorInvarianceImage = new unsigned char[imageSize];
    unsigned char* grayscaleImage = new unsigned char[width * height];
    unsigned char* UComponentImage = new unsigned char[width * height];

    // Copy the converted images back to host
    cudaMemcpy(colorInvarianceImage, d_colorInvarianceImage, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(grayscaleImage, d_grayscaleImage, width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(UComponentImage, d_UComponentImage, width * height, cudaMemcpyDeviceToHost);

    // Save the output images
    stbi_write_jpg(colorInvarianceOutputPath, width, height, 3, colorInvarianceImage, 100);
    stbi_write_jpg(grayscaleOutputPath, width, height, 1, grayscaleImage, 100);
    stbi_write_jpg(UComponentOutputPath, width, height, 1, UComponentImage, 100);

    // Cleanup
    stbi_image_free(inputImage);
    delete[] colorInvarianceImage;
    delete[] grayscaleImage;
    delete[] UComponentImage;
    cudaFree(d_inputImage);
    cudaFree(d_colorInvarianceImage);
    cudaFree(d_grayscaleImage);
    cudaFree(d_UComponentImage);

    return 0;
}

