#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// CUDA Kernel function to compute vector Normalization 
__global__ void vectorNorm(const float *vector_input, float *vector_norm, int power, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        //Sum each input in the vector raised to the power
        //This is equivalent to the Lp norm
        //||v||ₚ = (|v₁|ᵖ + |v₂|ᵖ + ... + |vₙ|ᵖ)^(1/p)
        //if you use simple pow(...) sometimes you'll get -nan
        atomicAdd(vector_norm, powf(vector_input[idx], power));
    }
}


//Function to generate a random float
float randomFloat(int randMax = 1000){
    //Return a random float between 0 and 1
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

// Main function
int main(){
    srand(time(0));
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVec, *hostNorm;
    float *deviceVector, *deviceNorm;
    int power = 2; //Change this to compute different norms (e.g., 1 for L1 norm, 2 for L2 norm)   
    int float_size = sizeof(float); 


    hostVec = (float*) malloc(size);
    hostNorm = (float*) malloc(sizeof(float)); 

    for(int idx = 0; idx < numElements; idx++){
        hostVec[idx] = randomFloat();
    }   

    cudaMalloc((void**)&deviceVector, size);    
    cudaMalloc((void**)&deviceNorm, sizeof(float));   //Result is a single float so no need to use sizeof(size)
    cudaMemset(deviceNorm, 0, sizeof(float));

    cudaMemcpy(deviceVector, hostVec, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    //Cell to vector normalization kernel
    vectorNorm<<<blocksPerGrid, threadsPerBlock>>>(deviceVector,
                                               deviceNorm,
                                               power,
                                               numElements);

    //Copy back & take the root
    cudaMemcpy(hostNorm, deviceNorm,
            sizeof(float), cudaMemcpyDeviceToHost);
    *hostNorm = pow(*hostNorm, 1.0f / power);

    std::cout << "Result is Vec Norm: " << *hostNorm << std::endl;

    // Free the memory allocated for gpu host vecs
    free(hostVec);
    free(hostNorm);

    // Free the memory allocated for real device vecs
    cudaFree(deviceVector);
    cudaFree(deviceNorm);
}