__global__ void colorConversionKernel(float* result, float* image_double, float* smoothmask, float* ratio_red, float* ratio_green, float* ratio_blue, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;

        float smooth_val = 1 - smoothmask[index];

        result[index] = (ratio_red[index] + 1) / (smooth_val * ratio_red[index] + 1) * image_double[index];
        result[index + width * height] = (ratio_green[index] + 1) / (smooth_val * ratio_green[index] + 1) * image_double[index + width * height];
        result[index + 2 * width * height] = (ratio_blue[index] + 1) / (smooth_val * ratio_blue[index] + 1) * image_double[index + 2 * width * height];
    }
}

void convertMatlabToCUDA(float* result, float* image_double, float* smoothmask, float* ratio_red, float* ratio_green, float* ratio_blue, int width, int height) {
    int size = width * height * sizeof(float);
    float *d_result, *d_image_double, *d_smoothmask, *d_ratio_red, *d_ratio_green, *d_ratio_blue;

    cudaMalloc(&d_result, 3 * size);
    cudaMalloc(&d_image_double, 3 * size);
    cudaMalloc(&d_smoothmask, size);
    cudaMalloc(&d_ratio_red, size);
    cudaMalloc(&d_ratio_green, size);
    cudaMalloc(&d_ratio_blue, size);

    cudaMemcpy(d_image_double, image_double, 3 * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_smoothmask, smoothmask, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratio_red, ratio_red, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratio_green, ratio_green, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratio_blue, ratio_blue, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
    colorConversionKernel<<<numBlocks, threadsPerBlock>>>(d_result, d_image_double, d_smoothmask, d_ratio_red, d_ratio_green, d_ratio_blue, width, height);

    cudaMemcpy(result, d_result, 3 * size, cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_image_double);
    cudaFree(d_smoothmask);
    cudaFree(d_ratio_red);
    cudaFree(d_ratio_green);
    cudaFree(d_ratio_blue);
}
