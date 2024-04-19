// so heres the individual parts i assume will solve this

//part 1

// Process 5.1.1 step 1 (part 1)
    // Inputs:
        // input: Original Image
            // Malloc and allocate the original input image into this
        // Mask: ShadowMask, Light mask
            // You'll need to run this for both the shadow and light mask.
        // Image width and height
    // Outputs:
        // Multiplied array of image and eroded mask
        // These are separated by channel Output[1]=red, etc, etc

// by the end of part 1, you should have this part of the function for the shadow and light rgb,
//lines 75-77 and 78-80, image_double(:,:,1).*eroded_gray_shadow_mask
// we only multiplied the two matrices together in this part


//part 2
// Process 5.1.2 step 2
//Overview: this is doing this part: sum(sum(eroded_gray_shadow_mask)) and the sum of the two parts above individually
    // Inputs:
        // array
            // You'll need to do this for the output in step 1. Since we do 
            //image_double(:,:,3).*eroded_gray_shadow_mask without the sum in the first thing: image_double(:,:,3).*eroded_gray_shadow_mask))
            // And you'll need to do this for the lighteroded_shadow_mask
    // Outputs:
        // Summed up array
// here you should have four outputs essentially
// you'll have the two outputs from step 1 summed, and two outputs which are just the shadow and light arrays summed up



// Process 5.1.3 step 3
// Do this on the host side, this is just division.
    // From step 1, you should have the following:
    // Output rgbs of image_double(:,:,3).*eroded_gray_shadow_mask for shadow and light sides. (So two output arrays)

    // From step 2, you should have the following
    // From those two outputs in step 1, you should have:
        // The array sum of the individual channels. IE output[1] is passed in to get the red sum, etc etc
        // You'll have thus one average value for outputs[1-3] rgb channels for both light and shadow
    // You should also have the sum of the eroded_shadow_mask and light eroded

    // Now, simply divide the output channel by the eroded shadow mask
    // IE in host: SummedOutput[1]/SummedErodedShadowMask = shadow_average_red

// really this part is just doing the division

// Process 5.2.1 step 3
    // ratio_red = litavg_red/shadowavg_red - 1;
    // Get all your ratios
    // this can be done in the host since it's just regular values at this point

// part 3
// prcess 5.3.1 step 4
    //inputs
        //the ratios you made
        //the smoothmask (not sure wehre this comes from)
        //the original image
    //output
        //the end of it all

//so admittedly everything above was gained with chatgpt, so like take what you will with that i just want to be done with all of this
// all that's needed is to piece it together on the host code and like you know ideally it works lmao





// Kernel to perform element-wise multiplication of two matrices for each RGB channel
__global__ void multiplyChannels(unsigned char *input, unsigned char *mask, unsigned char *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        // Compute index for each channel (RGB)
        int indexR = index;
        int indexG = index + width * height;
        int indexB = index + 2 * width * height;

        // Perform element-wise multiplication for each channel
        output[indexR] = input[indexR] * mask[index];
        output[indexG] = input[indexG] * mask[index];
        output[indexB] = input[indexB] * mask[index];
    }
}
// Process 5.1.1 step 1
    // Inputs:
        // input: Original Image
            // Malloc and allocate the original input image into this
        // Mask: ShadowMask, Light mask
            // You'll need to run this for both the shadow and light mask.
        // Image width and height
    // Outputs:
        // Multiplied array of image and eroded mask
        // These are separated by channel Output[1]=red, etc, etc


// CUDA kernel to perform parallel reduction
__global__ void sumArray(unsigned char *array, unsigned char *result, int N) {
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



// CUDA function to perform element-wise multiplication for each RGB channel
void process5(unsigned char *input, unsigned char *mask, unsigned char *output, int width, int height) {

    // ok first part in this function covers 5.1.1, fill in for light mask
    // maybe increase to a second output

    // Calculate total number of elements
    int numElements = width * height;

    // output arrray
    unsigned char *h_output_ShadowRGB = (float*)malloc(3 * width * height * sizeof(unsigned char)); // Allocate for RGB channels
    unsigned char *h_output_LightRGB = (float*)malloc(3 * width * height * sizeof(unsigned char)); // Allocate for RGB channels


    // Allocate device memory
    unsigned char *d_inputImage *d_image, *d_erodedMaskShadow, d_erodedMaskLight *d_channelMaskMult;
    cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char)); // delete, unneeded as we should have this
    cudaMalloc((void**)&d_erodedMaskShadow, grayscaleSize * sizeof(unsigned char)); // Light mask
    cudaMalloc((void**)&d_erodedMaskLight, grayscaleSize * sizeof(unsigned char)); // Shadow Mask


    // Our output
    cudaMalloc((void**)&d_channelMaskMult, imageSize * sizeof(unsigned char)); // Result of InputImg .* Mask

    // Copy input matrices to device memory
    cudaMemcpy(d_inputImage, image, 3 * numElements * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_erodedMaskShadow, mask, numElements * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel for our shadow mask first
    multiplyChannelsKernel<<<gridSize, blockSize>>>(d_image, d_erodedMaskShadow, d_channelMaskMult, width, height);

    // Copy result back to host
    cudaMemcpy(h_output_ShadowRGB, d_channelMaskMult, imageSize * sizeof(unsigned char)), cudaMemcpyDeviceToHost);

    // Launch kernel for light mask now
    // may need to reallocaste d_channelMaskMult
    multiplyChannelsKernel<<<gridSize, blockSize>>>(d_image, d_erodedMaskLight, d_channelMaskMult, width, height);
    cudaMemcpy(h_output_LightRGB, d_channelMaskMult, imageSize * sizeof(unsigned char)), cudaMemcpyDeviceToHost);

    // so now you should have the outputs to the multiplied shadow mask and original image rgb fields

//////////////////////////////////
     // Begin summing the arrays
    const int N = 1024; // Size of the array (will be the image we use, change to size of the image like ImageSize)
    const int blockSize = 256; // Threads per block
    const int numBlocks = (N + blockSize - 1) / blockSize; // Number of blocks needed

    // Allocate memory on the host for the array
    unsigned char *h_ChannelsMaskSum = new char[N]; // CONVERT TO OUR USE CASE
    unsigned char *h_ErodedMaskSum = new char[N]; // CONVERT TO OUR USE CASE

    // Copy array data from host to device
    // two for the shadow and light mask of each
    cudaMemcpy(d_ChannelsMask, h_ChannelsMaskSum, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ChannelsMask, h_ChannelsMaskSum, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // two for th
    cudaMemcpy(d_ErodedMaskS, h_ErodedMaskShadow, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ErodedMaskL, h_ErodedMaskLight, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    sumArray<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_erodedMaskShadow, d_result, N);
    sumArray<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_ChannelsMask, d_result, N);

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

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_channelMaskMult);
}

int main() {
    // Assuming you have already loaded the image and mask into arrays 'image' and 'mask' respectively

    // Dimensions of the image and mask
    int width = /* Width of the image */;
    int height = /* Height of the image */;

    // Allocate memory for output array
    float *output = (float*)malloc(3 * width * height * sizeof(float)); // Allocate for RGB channels

    // Call CUDA function
    multiplyRGBChannels(image, mask, output, width, height);

    // Free memory
    free(output);

    return 0;
}


