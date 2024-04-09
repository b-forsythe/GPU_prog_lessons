#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <limits.h>

#define SIZE 8000000
#define THREADS 8
#define BLOCKS 1


// begin findMinimum function, using CUDA to separate
__global__ void findMinimum(int *data, int *results, int size) {
    int tid = threadIdx.x;
    int step_size = size / blockDim.x;
    int start = tid * step_size;
    int end = start + step_size;

    int min = INT_MAX;
    for(int i = start; i < end; i++) {
        if(data[i] < min) min = data[i];
    }
    results[tid] = min;
}

int main() {
    int *data = (int*)malloc(SIZE * sizeof(int));
    int *results = (int*)malloc(THREADS * sizeof(int));
    
    // seed and set random values in array
    srand(time(NULL));
    for(int i = 0; i < SIZE; i++) {
        data[i] = rand() % 1000000000;
    }
    int *dev_data, *dev_results;
    cudaMalloc((void**)&dev_data, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_results, THREADS * sizeof(int));
    cudaMemcpy(dev_data, data, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // run findMinimum
    findMinimum<<<BLOCKS, THREADS>>>(dev_data, dev_results, SIZE);
    cudaMemcpy(results, dev_results, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    int min = INT_MAX;
    for(int i = 0; i < THREADS; i++) {
        if(results[i] < min) min = results[i];
    }

    printf("Parallel min: %d\n", min);

    // seq
    min = data[0];
    for(int i = 1; i < SIZE; i++) {
        if(data[i] < min) min = data[i];
    }

    printf("Sequential min: %d\n", min);

    // Clean up
    free(data); free(results);
    cudaFree(dev_data); cudaFree(dev_results);
    
    return 0;
}