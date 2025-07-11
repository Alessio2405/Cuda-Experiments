#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cassert>

constexpr int BLOCK_SIZE = 16;
constexpr int TILE_SIZE = 32;
constexpr float EPSILON = 1e-8f;

//Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

//Atomic max for float values
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}

//Optimized kernel with numerical stability and memory access
__global__ void fusedScaledDotProductAttentionOptimized(
    const float* __restrict__ Q, 
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ attention,
    int sequenceLength, 
    int dim,
    float scale
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Shared memory for intermediate computations since kernel could have concurrent threads asking for same resource
    __shared__ float sharedQ[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedK[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float maxVals[BLOCK_SIZE];
    __shared__ float sumExp[BLOCK_SIZE];
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        maxVals[threadIdx.y] = -INFINITY;
        sumExp[threadIdx.y] = 0.0f;
    }
    __syncthreads();
    
    if (row >= sequenceLength) return;
    
    float maxVal = -INFINITY;
    float sum = 0.0f;
    
    for (int tileIdx = 0; tileIdx < (sequenceLength + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
        int kCol = tileIdx * BLOCK_SIZE + threadIdx.x;
        
        if (kCol < sequenceLength) {
            float dotProduct = 0.0f;
            
            for (int dimTile = 0; dimTile < (dim + BLOCK_SIZE - 1) / BLOCK_SIZE; ++dimTile) {
                int dimIdx = dimTile * BLOCK_SIZE + threadIdx.x;
                
                // Load Q and K tiles into shared memory
                if (dimIdx < dim && row < sequenceLength) {
                    sharedQ[threadIdx.y][threadIdx.x] = Q[row * dim + dimIdx];
                } else {
                    sharedQ[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                if (dimIdx < dim && kCol < sequenceLength) {
                    sharedK[threadIdx.y][threadIdx.x] = K[kCol * dim + dimIdx];
                } else {
                    sharedK[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                __syncthreads();
                
                for (int k = 0; k < BLOCK_SIZE && dimTile * BLOCK_SIZE + k < dim; ++k) {
                    dotProduct += sharedQ[threadIdx.y][k] * sharedK[threadIdx.x][k];
                }
                
                __syncthreads();
            }
            
            dotProduct *= scale;
            maxVal = fmaxf(maxVal, dotProduct);
        }
    }
    
    atomicMaxFloat(&maxVals[threadIdx.y], maxVal);
    __syncthreads();
    
    for (int tileIdx = 0; tileIdx < (sequenceLength + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
        int kCol = tileIdx * BLOCK_SIZE + threadIdx.x;
        
        if (kCol < sequenceLength) {
            float dotProduct = 0.0f;
            
            for (int dimTile = 0; dimTile < (dim + BLOCK_SIZE - 1) / BLOCK_SIZE; ++dimTile) {
                int dimIdx = dimTile * BLOCK_SIZE + threadIdx.x;
                
                if (dimIdx < dim && row < sequenceLength) {
                    sharedQ[threadIdx.y][threadIdx.x] = Q[row * dim + dimIdx];
                } else {
                    sharedQ[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                if (dimIdx < dim && kCol < sequenceLength) {
                    sharedK[threadIdx.y][threadIdx.x] = K[kCol * dim + dimIdx];
                } else {
                    sharedK[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                __syncthreads();
                
                for (int k = 0; k < BLOCK_SIZE && dimTile * BLOCK_SIZE + k < dim; ++k) {
                    dotProduct += sharedQ[threadIdx.y][k] * sharedK[threadIdx.x][k];
                }
                
                __syncthreads();
            }
            
            dotProduct *= scale;
            float expVal = expf(dotProduct - maxVals[threadIdx.y]);
            sum += expVal;
        }
    }
    
    atomicAdd(&sumExp[threadIdx.y], sum);
    __syncthreads();
    
    if (col < dim) {
        float finalValue = 0.0f;
        
        for (int tileIdx = 0; tileIdx < (sequenceLength + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
            int kCol = tileIdx * BLOCK_SIZE + threadIdx.x;
            
            if (kCol < sequenceLength) {
                float dotProduct = 0.0f;
                
                // Recompute dot product
                for (int dimTile = 0; dimTile < (dim + BLOCK_SIZE - 1) / BLOCK_SIZE; ++dimTile) {
                    int dimIdx = dimTile * BLOCK_SIZE + threadIdx.x;
                    
                    if (dimIdx < dim && row < sequenceLength) {
                        sharedQ[threadIdx.y][threadIdx.x] = Q[row * dim + dimIdx];
                    } else {
                        sharedQ[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    
                    if (dimIdx < dim && kCol < sequenceLength) {
                        sharedK[threadIdx.y][threadIdx.x] = K[kCol * dim + dimIdx];
                    } else {
                        sharedK[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    
                    __syncthreads();
                    
                    for (int k = 0; k < BLOCK_SIZE && dimTile * BLOCK_SIZE + k < dim; ++k) {
                        dotProduct += sharedQ[threadIdx.y][k] * sharedK[threadIdx.x][k];
                    }
                    
                    __syncthreads();
                }
                
                dotProduct *= scale;
                float attentionWeight = expf(dotProduct - maxVals[threadIdx.y]) / (sumExp[threadIdx.y] + EPSILON);
                
                // Accumulate weighted values
                finalValue += attentionWeight * V[kCol * dim + col];
            }
        }
        
        attention[row * dim + col] = finalValue;
    }
}

void fillMatrixImproved(float* matrix, int rows, int cols, float mean = 0.0f, float stddev = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);
    
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

bool validateAttention(const float* attention, int sequenceLength, int dim) {
    for (int i = 0; i < sequenceLength * dim; ++i) {
        if (std::isnan(attention[i]) || std::isinf(attention[i])) {
            std::cerr << "Invalid attention value at index " << i << ": " << attention[i] << std::endl;
            return false;
        }
    }
    return true;
}

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

int main() {
    int sequenceLength = 512;
    int dim = 1024;
    float scale = 1.0f / sqrtf(static_cast<float>(dim));
    
    size_t size = sequenceLength * dim * sizeof(float);
    
    std::cout << "Enhanced Fused Scaled Dot-Product Attention" << std::endl;
    std::cout << "Sequence Length: " << sequenceLength << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Scale Factor: " << scale << std::endl;
    std::cout << "Memory per matrix: " << size / (1024 * 1024) << " MB" << std::endl;
    

    float *d_Q, *d_K, *d_V, *d_attention;
    CUDA_CHECK(cudaMalloc(&d_Q, size));
    CUDA_CHECK(cudaMalloc(&d_K, size));
    CUDA_CHECK(cudaMalloc(&d_V, size));
    CUDA_CHECK(cudaMalloc(&d_attention, size));
    

    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_attention = (float*)malloc(size);
    
    assert(h_Q && h_K && h_V && h_attention);


    std::cout << "Initializing matrices..." << std::endl;
    fillMatrixImproved(h_Q, sequenceLength, dim, 0.0f, 1.0f / sqrtf(dim));
    fillMatrixImproved(h_K, sequenceLength, dim, 0.0f, 1.0f / sqrtf(dim));
    fillMatrixImproved(h_V, sequenceLength, dim, 0.0f, 1.0f);
    
    Timer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice));
    std::cout << "Memory transfer to device: " << timer.elapsed() << " ms" << std::endl;
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (sequenceLength + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    std::cout << "Grid dimensions: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block dimensions: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    
    fusedScaledDotProductAttentionOptimized<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_attention, sequenceLength, dim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const int numRuns = 10;
    timer.start();
    
    for (int run = 0; run < numRuns; ++run) {
        fusedScaledDotProductAttentionOptimized<<<gridDim, blockDim>>>(
            d_Q, d_K, d_V, d_attention, sequenceLength, dim, scale
        );
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    double totalTime = timer.elapsed();
    double avgTime = totalTime / numRuns;
    
    std::cout << "Average kernel execution time: " << avgTime << " ms" << std::endl;
    
    long long ops = (long long)sequenceLength * sequenceLength * dim * 2; 
    double gflops = (ops / 1e9) / (avgTime / 1000.0);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    timer.start();
    CUDA_CHECK(cudaMemcpy(h_attention, d_attention, size, cudaMemcpyDeviceToHost));
    std::cout << "Memory transfer to host: " << timer.elapsed() << " ms" << std::endl;
    
    std::cout << "Validating results..." << std::endl;
    if (validateAttention(h_attention, sequenceLength, dim)) {
        std::cout << "✓ Attention computation successful!" << std::endl;
        
        std::cout << "Sample attention values:" << std::endl;
        for (int i = 0; i < std::min(5, sequenceLength); ++i) {
            std::cout << "Row " << i << ": ";
            for (int j = 0; j < std::min(5, dim); ++j) {
                std::cout << h_attention[i * dim + j] << " ";
            }
            std::cout << "..." << std::endl;
        }
    } else {
        std::cerr << "✗ Attention computation failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_attention));
    
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_attention);
    
    std::cout << "Enhanced attention computation completed successfully!" << std::endl;
    
    return 0;
}