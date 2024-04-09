#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define N 8

// create block instead of 2d arr
__global__ void blockSumKernel(int *d_in, int *d_out) {
    extern __shared__ int sharedMem[];
    
    int tid = threadIdx.x;
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // send to shared mem
    sharedMem[tid] = d_in[blockId];
    __syncthreads(); 
    
    // reduce shared mem
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads(); // sync to ensure proper data
    }
    
    if(tid == 0) {
        d_out[blockIdx.x] = sharedMem[0];
    }
}

int main() {
    int h_in[N][N], h_out[N];
    int *d_in, *d_out;

    // create array and randomize values
    srand(time(NULL));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            h_in[i][j] = rand() % 10;
        }
    }

    cudaMalloc((void**)&d_in, N * N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * N * sizeof(int), cudaMemcpyHostToDevice);

    blockSumKernel<<<N, N, N * sizeof(int)>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute sum of all rows
    int totalSum = 0;
    for(int i = 0; i < N; i++)
        totalSum += h_out[i];

    printf("Total sum of all elements in the 2D array: %d\n", totalSum);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}