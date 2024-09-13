#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for parallel reduction (sum)
__global__ void reduceSum(float *input, float *output, int N) {
    extern __shared__ float shared_data[];

    // Each thread loads one element into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        shared_data[tid] = input[i];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() {
    int N = 1 << 20;  // Vector size (1M elements)
    size_t size = N * sizeof(float);

    // Allocate input vector in host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size / 1024);  // Temporary output

    // Initialize the input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Set all elements to 1 for simplicity
    }

    // Allocate vectors in device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size / 1024);

    // Copy input vector from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Set up execution configuration
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch kernel to perform reduction on the input vector
    reduceSum<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);

    // Copy output vector from device to host (this contains partial sums)
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform final reduction on the host to get the total sum
    float total_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        total_sum += h_output[i];
    }

    // Output the result
    printf("Total sum: %f\n", total_sum);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
