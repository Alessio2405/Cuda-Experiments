#include <iostream> // for std::cout
#include <cuda_runtime.h> // for CUDA runtime functions
#include <device_launch_parameters.h> // for CUDA kernel launch parameters


__global__ void vectorAdd(const int *a, 
                          const int *b, 
                          int *c, 
                          int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements){
        c[i] = a[i] + b[i];
    }
}

int main() {
    int numElements = 50;

    size_t size = numElements * sizeof(int);
    int *hostA, *hostB, *hostC;
    int *deviceA, *deviceB, *deviceC;
    hostA = (int*) malloc(size);
    hostB = (int*) malloc(size);
    hostC = (int*) malloc(size);

    for (int i = 0; i < numElements; i++){
        hostA[i] = i;
        hostB[i] = i*2;
    }

    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, size);

    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements; i++){
        std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << std::endl;
    }

    free(hostA);
    free(hostB);
    free(hostC);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}