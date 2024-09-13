#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK_ERROR(call)                                \
    {                                                         \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    }

int main() {
    // Matrix dimensions
    int m = 2;  // Number of rows in A and C
    int n = 3;  // Number of columns in B and C
    int k = 4;  // Number of columns in A and rows in B

    // Scalars
    float alpha = 1.0f;
    float beta = 0.0f;

    // Host matrices (row-major order)
    float h_A[m * k] = {1, 2, 3, 4, 
                        5, 6, 7, 8};  // 2x4 matrix
    float h_B[k * n] = {1, 2, 3, 
                        4, 5, 6, 
                        7, 8, 9, 
                        10, 11, 12};  // 4x3 matrix
    float h_C[m * n] = {0, 0, 0, 
                        0, 0, 0};  // 2x3 matrix

    // Device matrices
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_A, m * k * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_B, k * n * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_C, m * n * sizeof(float)));

    // Copy host matrices to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k, 
                &alpha, 
                d_B, n, 
                d_A, k, 
                &beta, 
                d_C, n);

    // Copy result matrix back to host
    CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Result matrix C (m x n):\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // Clean up
    cublasDestroy(handle);
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));

    return 0;
}
