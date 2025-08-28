#include <cstdio>

#include "matmul.h"
#define BLOCKS 4

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;
static cudaStream_t data_s, calc_s;
static cudaEvent_t events[BLOCKS];

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}

__global__ void gpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // int i = blockDim.x * blockIdx.x + threadIdx.x;
  // int j = blockDim.y * blockIdx.y + threadIdx.y;
  // if(i>=M || j>=N) return;
  
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int i = tid/N;
  int j = tid%N;

  float sum = 0.0;
  for(int k = 0; k < K; k+=4){
    float4 a4 = *(float4*)(&_A[i*K + k]);

    for(int l = 0; l < 4; l++){
      sum += ((float*)(&a4))[l] * _B[(k+l) * N + j];
    }

    // sum += a4.x * _B[k * N + j];
    // sum += a4.y * _B[(k+1) * N + j];
    // sum += a4.z * _B[(k+2) * N + j];
    // sum += a4.w * _B[(k+3) * N + j];
  }
  _C[i * N + j] = sum;
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  // naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  int Mbegin[BLOCKS],Mend[BLOCKS];
  for(int i=0; i<BLOCKS; i++){
    Mbegin[i] = M / BLOCKS * i;
    Mend[i] = M / BLOCKS * (i+1);
    if(i == BLOCKS-1) Mend[i] = M;
  }

  CHECK_CUDA(cudaMemcpyAsync(B_gpu, _B, K*N*sizeof(float),cudaMemcpyHostToDevice, data_s));
  for(int i=0; i<BLOCKS; i++){
    CHECK_CUDA(cudaMemcpyAsync(&A_gpu[Mbegin[i]*K], &_A[Mbegin[i]*K], (Mend[i]-Mbegin[i])*K*sizeof(float), cudaMemcpyHostToDevice, data_s));
    CHECK_CUDA(cudaEventRecord(events[i],data_s));
  }
  
  // (TODO) Launch kernel on a GPU

  for(int i=0; i<BLOCKS; i++){
    dim3 gridDim(((Mend[i]-Mbegin[i])*N+1023)/1024);
    dim3 blockDim(1024);
    cudaStreamWaitEvent(calc_s, events[i]);
    gpu_matmul<<<gridDim, blockDim, 0, calc_s>>>(&A_gpu[Mbegin[i]*K], B_gpu, &C_gpu[Mbegin[i]*N], (Mend[i]-Mbegin[i]), N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  // (TODO) Download C matrix from GPU
  cudaStreamSynchronize(calc_s);
  CHECK_CUDA(cudaMemcpyAsync(_C, C_gpu, M*N*sizeof(float),cudaMemcpyDeviceToHost, data_s));
  cudaStreamSynchronize(data_s);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaStreamCreate(&data_s));
  CHECK_CUDA(cudaStreamCreate(&calc_s));
  for(int i=0; i<BLOCKS; i++){
    CHECK_CUDA(cudaEventCreate(&events[i]));
  }
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, N*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M*N*sizeof(float)));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
