#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define SIZE 8000000

int main(int argc, char** argv) {
    int rank, size, min;
    int *data = NULL, *chunk;
    int chunk_size;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk_size = SIZE / size;
    chunk = (int*)malloc(chunk_size * sizeof(int));
if (rank == 0) {
    // Process 0 generates array
    data = (int*)malloc(SIZE * sizeof(int));

    // Seed the random number generator with current time, ensuring different sequences each run
    srand(time(NULL));

    for(int i=0; i<SIZE; i++) {
        data[i] = rand() % 1000000000;
    }
}

    // Scatter the data array
    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each Process finds the minimum of its chunk
    min = INT_MAX;
    for(int i=0; i<chunk_size; i++) {
        if (chunk[i] < min) min = chunk[i];
    }

    // Send the minimum back to process 0
    if(rank != 0) {
        MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        int global_min = min;
        for(int i=1; i<size; i++) {
            MPI_Recv(&min, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            if (min < global_min) global_min = min;
        }
        printf("Parallel min: %d\n", global_min);

        // Validate this answer by finding the min sequentially.
        min = data[0];
        for (int i=0; i<SIZE; i++) {
            if (data[i] < min) min = data[i];
        }
        printf("Sequential min: %d\n", min);
    }

    // Free allocated memory
    if (rank == 0) free(data);
    free(chunk);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}