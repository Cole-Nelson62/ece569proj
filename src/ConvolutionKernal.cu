

__global__ void convolution_basic_kernel(float *input, float *Mask, float *output, int Mask_Width, int Width, int Height) 
{
    //__constant__ float floatConst[15];



    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row =blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < Width && Row < Height) 
    { // boundary checking for each output element
        int pixVal = 0; // initialize local product value to 0. 
        int start_col = Col-(Mask_Width/2);
        int start_row = Row-(Mask_Width/2);
        // Get the of the surrounding box
        for(int j = 0; j < Mask_Width; ++j) { // row 
            for(int k = 0; k < Mask_Width; ++k) { //column
                int curRow =start_row+j;
                int curCol = start_col+k;
                // Verify we have a valid image pixel
                if((curRow > -1 && curRow < Height) && (curCol > -1 && curCol < Width) ) {
                    pixVal += input[curRow * Width + curCol] * Mask[j * Mask_Width + k ];
                } 
        } 
    }
    // Write our new pixel value out
    output[Row* Width + Col] = (unsigned char)(pixVal);
    } 
}
/*
//currently same
__global__ void convolution_tiling(float *N, float *M, float *P, int Mask_Width, int Width, int Height) 
{
    __constant__ float floatConst[15];



    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row =blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) 
    { // boundary checking for each output element
        int pixVal = 0; // initialize local product value to 0. 
        start_col = col-(Mask_Width/2);
        start_row = row-(Mask_Width/2);
        // Get the of the surrounding box
        for(int j = 0; j < maskwidth; ++j) { // row 
            for(int k = 0; k < maskwidth; ++k) { //column
                int curRow =start_row+j;
                int curCol = start_col+k;
                // Verify we have a valid image pixel
                if((curRow > -1 && curRow < Height) && (curCol > -1 && curCol < Width) ) {
                    pixVal += in[curRow * Width + curCol] * mask[j * Mask_Width + k ];
                } 
        } 
    }
    // Write our new pixel value out
    out[Row* Width + Col] = (unsigned char)(pixVal);
    } 
}

//currently same
__global__ void convolution_Optimized(float *N, float *M, float *P, int Mask_Width, int Width, int Height) 
{
    __constant__ float floatConst[15];



    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row =blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) 
    { // boundary checking for each output element
        int pixVal = 0; // initialize local product value to 0. 
        start_col = col-(Mask_Width/2);
        start_row = row-(Mask_Width/2);
        // Get the of the surrounding box
        for(int j = 0; j < maskwidth; ++j) { // row 
            for(int k = 0; k < maskwidth; ++k) { //column
                int curRow =start_row+j;
                int curCol = start_col+k;
                // Verify we have a valid image pixel
                if((curRow > -1 && curRow < Height) && (curCol > -1 && curCol < Width) ) {
                    pixVal += in[curRow * Width + curCol] * mask[j * Mask_Width + k ];
                } 
        } 
    }
    // Write our new pixel value out
    out[Row* Width + Col] = (unsigned char)(pixVal);
    } 
}
*/