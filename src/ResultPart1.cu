// Kernel to perform element-wise multiplication of two matrices for each RGB channel
__global__ void multiplyChannelsKernel(const float *input, const float *mask, float *output, int width, int height) {
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

// CUDA function to perform element-wise multiplication for each RGB channel
void multiplyRGBChannels(const float *image, const float *mask, float *output, int width, int height) {
    // Calculate total number of elements
    int numElements = width * height;

    // Allocate device memory
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, 3 * numElements * sizeof(float)); // Allocate for RGB channels
    cudaMalloc(&d_mask, numElements * sizeof(float));
    cudaMalloc(&d_output, 3 * numElements * sizeof(float)); // Allocate for RGB channels

    // Copy input matrices to device memory
    cudaMemcpy(d_image, image, 3 * numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    multiplyChannelsKernel<<<gridSize, blockSize>>>(d_image, d_mask, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(output, d_output, 3 * numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
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