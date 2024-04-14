__global__ void sumAverageChannel(unsigned char* input, unsigned char* originalImg, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < width && y < height) {
        int pixelIndex = y * width + x;
        int rgbIndex = pixelIndex * 3;

        float R = input[rgbIndex];
        float G = input[rgbIndex + 1];
        float B = input[rgbIndex + 2];

        //outputU[pixelIndex] = static_cast<unsigned char>(128 + (-0.147 * R - 0.289 * G + 0.436 * B));
    }
}

__global__ void Erosion(unsigned char * inputImg, unsigned char* outImg, int width, int height, int strel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( y < height || x < width){

        unsigned int start_i = max(y - strel, 0);
        unsigned int end_i = min(height - 1, y + strel);
        unsigned int start_j = max(x - strel, 0);
        unsigned int end_j = min(width - 1, x + strel);
        int value = 255;
        for (int i = start_i; i <= end_i; i++) {
            for (int j = start_j; j <= end_j; j++) {
                value = min(value, inputImg[i * width + j]);
            }
        }
    outImg[y * width + x] = value;

    }
}

