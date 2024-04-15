#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "shadowRemoval.cu"
#include "ConvolutionKernal.cu"
#include "Erosion.cu"
#include <wb.h>


#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <inputImagePath> <colorInvarianceOutputPath> <grayscaleOutputPath> <UComponentOutputPath>" << std::endl;
        return -1;
    }

    // For calculating proccesses
    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);
    
    // for total time
    cudaEvent_t atotalStartEvent, atotalStopEvent;
    float atotalElapsedTime;
    cudaEventCreate(&atotalStartEvent);
    cudaEventCreate(&atotalStopEvent);

    
    //if (argc < 5) {
    //    std::cerr << "Usage: " << argv[0] << " <inputImagePath> <colorInvarianceOutputPath> <grayscaleOutputPath> <UComponentOutputPath>" << std::endl;
    //    return -1;
    //}



    const char* inputImagePath = argv[1];
    const char* colorInvarianceOutputPath = argv[2];
    const char* grayscaleOutputPath = argv[3];
    const char* UComponentOutputPath = argv[4];
    const char* ConvoOutputPath = argv[5];
    const char* ErodedLightOutputPath = argv[6];
    const char* ErodedShadowOutputPath  = argv[7];
    const char* FinalOutputPath = argv[8];

    int width, height, channels;
    unsigned char* inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);
     int Mask_Width;

    if (inputImage == NULL) {
        std::cerr << "Error loading image: " << inputImagePath << std::endl;
        return -1;
    }

    if (channels < 3) {
        std::cerr << "Error: Image must have at least 3 channels (RGB)" << std::endl;
        stbi_image_free(inputImage);
        return -1;
    }

    cudaEventRecord(atotalStartEvent, 0);

    int imageSize = width * height * channels;
    int grayscaleSize = width * height;

    unsigned char *d_inputImage, *d_colorInvarianceImage, *d_grayscaleImage, *d_UComponentImage, *d_GreyScaleMask, *d_YUVMask, *d_ConvoOutput;
    unsigned char *d_greyscalethreshold, *d_yuvthreshold;

    wbTime_start(GPU, "Copying input memory to the GPU.");

    cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_colorInvarianceImage, imageSize * sizeof(unsigned char)); // 3 channels
    cudaMalloc((void**)&d_grayscaleImage, grayscaleSize * sizeof(unsigned char)); // Single channel
    cudaMalloc((void**)&d_UComponentImage, grayscaleSize * sizeof(unsigned char)); // U component
    //cudaMalloc((void**)&d_UComponentImage, grayscaleSize * sizeof(unsigned char)); // U component
    cudaMalloc((void**)&d_ConvoOutput, grayscaleSize * sizeof(unsigned char)); // U component

    cudaMalloc((void**)&d_GreyScaleMask, grayscaleSize * sizeof(unsigned char)); // greymask 
    cudaMalloc((void**)&d_YUVMask, grayscaleSize * sizeof(unsigned char)); // YUV Mask

    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    cudaEventRecord(astartEvent, 0);
    // Launch the combined kernel
    ColorTransformation<<<gridDim, blockDim>>>(d_inputImage, d_colorInvarianceImage, d_grayscaleImage, d_UComponentImage, width, height);

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\n");
    printf("Total time for Color transforms Proccess 1 (ms) %f \n",aelapsedTime);
    printf("\n");


    cudaEventRecord(astartEvent, 0);
    // Allocate memory for the histogram
    unsigned int* d_histogram;
    cudaMalloc((void**)&d_histogram, NUM_BINS * sizeof(unsigned int));
    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));

    // Configure kernel launch parameters
    int threadsPerBlock = 256;  // This is a typical choice; adjust based on GPU
    int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    // Launch histogram kernel for grayscale image
    computeHistogram<<<numBlocks, threadsPerBlock, NUM_BINS * sizeof(unsigned int)>>>(d_grayscaleImage, d_histogram, width * height);
    //calculateOtsuThreshold <<<numBlocks, threadsPerBlock, NUM_BINS * sizeof(unsigned int)>>>(d_histogram,imageSize,d_greyscalethreshold);
    // Launch histogram kernel for yuv image
    computeHistogram<<<numBlocks, threadsPerBlock, NUM_BINS * sizeof(unsigned int)>>>(d_UComponentImage, d_histogram, width * height);

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\n");
    printf("Total time for Proccess 2 Otsu (ms) %f \n",aelapsedTime);
    printf("\n");

    // for proccess 3 convolution
    cudaEventRecord(astartEvent, 0);
    Mask_Width =  11;
    convolution_basic_kernel<<<gridDim, blockDim>>>(d_UComponentImage, d_YUVMask, d_ConvoOutput, Mask_Width, width, height) ;

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\n");
    printf("Total time for Proccess 3 convolutions (ms) %f \n",aelapsedTime);
    printf("\n");

     // for proccess 4
    cudaEventRecord(astartEvent, 0);
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
    dim3 erodeGrid(ceil((float)width/erodeBlock.x), ceil((float)height / erodeBlock.y));

    Erosion<<<erodeGrid, erodeBlock>>>(d_GreyScaleMask, d_erodedMaskShadow, width, height, 2);
    //finish erases

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\n");
    printf("Total time for Proccess 4 errosion (ms) %f \n",aelapsedTime);
    printf("\n");


     // for proccess 5
    cudaEventRecord(astartEvent, 0);


     cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\n");
    printf("Total time for Proccess 5 result (ms) %f \n",aelapsedTime);
    printf("\n");


    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(atotalStopEvent);
    cudaEventElapsedTime(&atotalElapsedTime, atotalStartEvent, atotalStopEvent);
    printf("\n");
    printf("Total compute time of function after proccess 5 commits(ms) %f \n",aelapsedTime);
    printf("\n");


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
    unsigned char* ConvoOutput = new unsigned char[grayscaleSize];
    unsigned char* ErodedLight = new unsigned char[grayscaleSize];
    unsigned char* ErodedShadow = new unsigned char[grayscaleSize];
    unsigned char* Final = new unsigned char[imageSize];




    // Copy the converted images back to host
    cudaMemcpy(colorInvarianceImage, d_colorInvarianceImage, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(grayscaleImage, d_grayscaleImage, grayscaleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(UComponentImage, d_UComponentImage, grayscaleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ConvoOutput, d_ConvoOutput, grayscaleSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(ErodedLight, d_erodedMaskShadow, grayscaleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ErodedShadow, d_erodedMaskLight, grayscaleSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Final, d_ConvoOutput, imageSize, cudaMemcpyDeviceToHost);


    // Save the output images
    stbi_write_jpg(colorInvarianceOutputPath, width, height, 3, colorInvarianceImage, 100);
    stbi_write_jpg(grayscaleOutputPath, width, height, 1, grayscaleImage, 100);
    stbi_write_jpg(UComponentOutputPath, width, height, 1, UComponentImage, 100);
    stbi_write_jpg(ConvoOutputPath, width, height, 1, ConvoOutput, 100);

    stbi_write_jpg(ErodedLightOutputPath, width, height, 1, ErodedLight, 100);
    stbi_write_jpg(ErodedShadowOutputPath, width, height, 1, ErodedShadow, 100);
    //stbi_write_jpg(FinalOutputPath, width, height, 1, Final, 100);


    // Cleanup
    stbi_image_free(inputImage);
    delete[] colorInvarianceImage;
    delete[] grayscaleImage;
    delete[] UComponentImage;
    delete[] d_ConvoOutput;
    delete[] histogram;
    delete[] d_GreyScaleMask;
    delete[] d_YUVMask;
    delete[] d_erodedMaskLight;
    delete[] d_erodedMaskLight;

    cudaFree(d_inputImage);
    cudaFree(d_colorInvarianceImage);
    cudaFree(d_grayscaleImage);
    cudaFree(d_UComponentImage);
    cudaFree(d_ConvoOutput);
    cudaFree(d_histogram);
    cudaFree(d_GreyScaleMask);
    cudaFree(d_YUVMask);
    cudaFree(d_erodedMaskLight);
    cudaFree(d_erodedMaskLight);


    return 0;
}

