#include "utils.h"

__global__ void convolution_basic_kernel(unsigned char *input, unsigned char *Mask, unsigned char *output, int Mask_Width, int Width, int Height) 
{
    //__constant__ float floatConst[15];



    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row =blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < Width && Row < Height) 
    { // boundary checking for each output element
        unsigned char pixVal = 0; // initialize local product value to 0. 
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


__global__ void convolution_Shared_Mem(unsigned char *input, unsigned char *Mask, unsigned char *output, int Mask_Width, int Width, int Height) 
{
    //__constant__ float floatConst[15];
    __shared__ unsigned char convShared[BLOCK_SIZE][BLOCK_SIZE];


    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    int Col_i = Col - SHIFTS;
    int Row_I = Row - SHIFTS;
    //if ((Row_I >= 0 && Row_I < input.height) && (Col_i >= 0 && Col_i < input.width)) // to capture the A matrix
    if ((Row_I >= 0 && Row_I < Height) && (Col_i >= 0 && Col_i < Width)) // to capture the A matrix  
      {
        convShared[ty][tx] = input[Row_I * Width + Col_i];
      }
      else
      {
        convShared[ty][tx] = 0.0;
      }
     __syncthreads();

    if (Col < TILE_WIDTH && Row < TILE_WIDTH) 
    { // boundary checking for each output element
        unsigned char pixVal = 0; // initialize local product value to 0. 
        int start_col = Col-(Mask_Width/2);
        int start_row = Row-(Mask_Width/2);
        // Get the of the surrounding box
        for(int j = 0; j < Mask_Width; ++j) { // row 
            for(int k = 0; k < Mask_Width; ++k) { //column
                int curRow =start_row+j;
                int curCol = start_col+k;
                // Verify we have a valid image pixel
                if((curRow > -1 && curRow < Height) && (curCol > -1 && curCol < Width) ) {
                    //unsigned char *p1 = convShared[curRow * Width + curCol] ;
                    pixVal += convShared[ty + j][tx + k] * Mask[j * Mask_Width + k];
                } 
        } 
    }
    // Write our new pixel value out
    output[Row* Width + Col] = (unsigned char)(pixVal);
    } 
}


__global__ void convolution_seperateRow(unsigned char *input, unsigned char *Mask, unsigned char *output, int Mask_Width, int Width, int Height) 
{
        //__constant__ float floatConst[15];
    //__shared__ unsigned char convSharedRow[BLOCK_SIZE][TILE_WIDTH];
    __shared__ unsigned char convSharedRow[BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int by = blockDim.y;


    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    const int loc = Col + Row*Width;

    if (Col >= Width || Row >= Height) 
    {
        return;
    }
    /*
    //left side of the kernal 
    x = x0 - KERNEL_RADIUS;
    if ( Col < 0 ){
        convSharedRow[tx][ty] = 0;

    }  
    else
    {
        convSharedRow[threadIdx.x][threadIdx.y] = input[Row_I * Width + Col_i];
    }
    // Right side of the kernal
    int x = x0 - KERNEL_RADIUS;
    if ( Col < Width-1 ){
        convSharedRow[tx][ty] = 0;

    }  
    else
    {
        convSharedRow[threadIdx.x][threadIdx.y] = input[ Row_I * Width + Col_i];
    }
    */

for (int i = -bx; i <= bx; i+= bx) {
    int x0 = tx+ i;
    int newLoc = loc + i;
    if (x0 < -Mask_Width ||  x0 >= Mask_Width + bx ||newLoc < 0 || newLoc >= Width*Height)
        continue;
    convSharedRow[tx + i + BLOCK_SIZE + (ty) *(bx+(SHIFTScol))] = input[newLoc];
    }
    
    __syncthreads();
    for (int i = -Mask_Width; i <= Mask_Width; i++) {
        for (int j = -Mask_Width; j <= Mask_Width; j++) {
            unsigned int t = convSharedRow[ty+ i + Mask_Width + (ty+ j + Mask_Width)*(bx+(Mask_Width << 1))];
            int temp = output[i + Mask_Width +  (j+Mask_Width)*((Mask_Width << 1) + 1)];
            value = d_uintToRGB(t);
            value *= temp; 
            accumulation += value;
        }
    }
    output[Row* Width + Col] = (unsigned char)(pixVal);

}

__global__ void convolution_seperateCol(float *N, float *M, float *P, int Mask_Width, int Width, int Height) 
{
    //__constant__ float floatConst[15];
    __shared__ unsigned char convSharedCol[BLOCK_SIZE][BLOCK_SIZE];


    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

}
/*
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