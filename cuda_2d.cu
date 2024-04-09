#include <stdio.h>
#include <cuda.h>

#define N 3
#define M 4

//begin function to sum all values in a kernel
__global__ void colSumKernel(int *d_in, int *d_out) {
    int j = threadIdx.x;
    d_out[j] = 0;
    for(int i = 0; i < N; i++)
        d_out[j] += d_in[i * M + j];
}

int main() {
    int h_in[N][M], h_out[M];
    int *d_in, *d_out;

    // Create 2D array with random values
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            h_in[i][j] = rand() % 10;
        }
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&d_in, N * M * sizeof(int));
    cudaMalloc((void**)&d_out, M * sizeof(int));

    cudaMemcpy(d_in, h_in, N * M * sizeof(int), cudaMemcpyHostToDevice);

    colSumKernel<<<1, M>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, M * sizeof(int), cudaMemcpyDeviceToHost);

    // sum all columns
    int totalSum = 0;
    for(int i = 0; i < M; i++)
        totalSum += h_out[i];

    printf("Total sum of all elements in the 2D array: %d\n", totalSum);

    // free
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}