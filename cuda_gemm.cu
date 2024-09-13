#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA kernel for matrix multiplication (GEMM)
__global__ void matrixMul(float *A, float *B, float *C, int N, int M, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;

    if (row < N && col < K) {
        for (int e = 0; e < M; ++e) {
            value += A[row * M + e] * B[e * K + col];
        }
        C[row * K + col] = alpha * value + beta * C[row * K + col];
    }
}

int main() {
    int N = 10;  // Number of rows in A and C
    int M = 10;  // Number of columns in A and rows in B
    int K = 10;  // Number of columns in B and C

    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate host memory for matrices A, B, and C
    size_t size_A = N * M * sizeof(float);
    size_t size_B = M * K * sizeof(float);
    size_t size_C = N * K * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialize matrices A and B
    for (int i = 0; i < N * M; ++i) {
        h_A[i] = rand() % 100 / 100.0f;
    }
    for (int i = 0; i < M * K; ++i) {
        h_B[i] = rand() % 100 / 100.0f;
    }

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define the block and grid size
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the matrix multiplication kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, M, K, alpha, beta);

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify the result (optional, can be time-consuming for large matrices)

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            printf("%f ", h_C[i * K + j]);
        }
        printf("\n");
    }


    printf("Matrix multiplication completed successfully!\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
