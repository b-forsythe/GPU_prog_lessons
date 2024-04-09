#include <curand_kernel.h>
#include "FreeImage.h"
#include "stdio.h"
#include <time.h>

#define DIM 2048
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 80
#define THREADS_PER_BLOCK 512

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
  
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius)
        {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

__global__ void drawSpheres_GPU(Sphere* s, char *red, char *green, char *blue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Map idx to 2D space
    int x = idx % DIM;
    int y = (idx / DIM) % DIM;

    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
    for(int i=0; i<SPHERES; i++)
    {
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz)
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    red[idx] = (char) (r * 255);
    green[idx] = (char) (g * 255);
    blue[idx] = (char) (b * 255);
}

int main()
{
    FreeImage_Initialise();
    atexit(FreeImage_DeInitialise);
    FIBITMAP * bitmap = FreeImage_Allocate(DIM, DIM, 24);
    srand(time(NULL));

    char *red = (char *) malloc(DIM*DIM*sizeof(char));
    char *green = (char *) malloc(DIM*DIM*sizeof(char));
    char *blue = (char *) malloc(DIM*DIM*sizeof(char));

    char *red_gpu, *green_gpu, *blue_gpu;

    // Allocate memory for device
    cudaMalloc((void**)&red_gpu, DIM*DIM*sizeof(char));
    cudaMalloc((void**)&green_gpu, DIM*DIM*sizeof(char));
    cudaMalloc((void**)&blue_gpu, DIM*DIM*sizeof(char));

    Sphere spheres[SPHERES];
    for (int i = 0; i<SPHERES; i++)
    {
        spheres[i].r = rnd( 1.0f );
        spheres[i].g = rnd( 1.0f );
        spheres[i].b = rnd( 1.0f );
        spheres[i].x = rnd( (float) DIM ) - DIM/2.0;
        spheres[i].y = rnd( (float) DIM ) - DIM/2.0;
        spheres[i].z = rnd( (float) DIM ) - DIM/2.0;
        spheres[i].radius = rnd( 200.0f ) + 40;
    }

    Sphere* sphere_gpu;
    cudaMalloc((void**)&sphere_gpu, SPHERES*sizeof(Sphere));
    cudaMemcpy(sphere_gpu, spheres, SPHERES*sizeof(Sphere), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blocks(DIM*DIM/THREADS_PER_BLOCK,1,1);
    dim3 threads(THREADS_PER_BLOCK,1,1);

    // Call the kernel
    drawSpheres_GPU<<<blocks, threads>>>(sphere_gpu, red_gpu, green_gpu, blue_gpu);

    // Copy data back from GPU to CPU
    cudaMemcpy(red, red_gpu, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(green, green_gpu, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(blue, blue_gpu, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);

    RGBQUAD color;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            int index = j*DIM + i;
            color.rgbRed = red[index];
            color.rgbGreen = green[index];
            color.rgbBlue = blue[index];
            FreeImage_SetPixelColor(bitmap, i, j, &color);
        }
    }

    FreeImage_Save(FIF_PNG, bitmap, "rayGPU.png", 0);
    FreeImage_Unload(bitmap);

    cudaFree(red_gpu);
    cudaFree(green_gpu);
    cudaFree(blue_gpu);
    cudaFree(sphere_gpu);

    free(red);
    free(green);
    free(blue);

    return 0;
}