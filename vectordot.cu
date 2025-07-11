#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Kernel function to compute the dot product of two vectors
__global__ void computeDotProduct(const float *vectorFirst, const float *vectorSecond, float *dotProductResult, int vectorSize) {
    int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIndex < vectorSize) {
        atomicAdd(dotProductResult, vectorFirst[threadIndex] * vectorSecond[threadIndex]);
    }
}

// Function to generate a random float
float generateRandomFloat(int maxValue = 1000) {
    return static_cast<float>(rand()) / static_cast<float>(maxValue);
}

int main() {
    srand(time(0));

    int vectorSize = 300;
    size_t memorySize = vectorSize * sizeof(float);

    float *hostVectorA, *hostVectorB, *hostResult;
    float *deviceVectorA, *deviceVectorB, *deviceResult;

    hostVectorA = (float*) malloc(memorySize);
    hostVectorB = (float*) malloc(memorySize);
    hostResult = (float*) malloc(sizeof(float));

    for (int index = 0; index < vectorSize; index++) {
        hostVectorA[index] = generateRandomFloat();
        hostVectorB[index] = generateRandomFloat();
    }

    cudaMalloc((void**)&deviceVectorA, memorySize);
    cudaMalloc((void**)&deviceVectorB, memorySize);
    cudaMalloc((void**)&deviceResult, sizeof(float));

    cudaMemcpy(deviceVectorA, hostVectorA, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, vectorSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    computeDotProduct<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceVectorB, deviceResult, vectorSize);

    cudaMemcpy(hostResult, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Dot product: " << *hostResult << std::endl;

    free(hostVectorA);
    free(hostVectorB);
    free(hostResult);

    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceResult);
}
