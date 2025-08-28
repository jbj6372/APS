#include <cstdio>

#include "convolution.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

static float *I_gpu, *F_gpu, *O_gpu;
cudaStream_t data_s, calc_s;
cudaEvent_t events[100];

void naive_cpu_convolution(float *_I, float *_F, float *_O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  float *I = _I, *F = _F, *O = _O;
  // Naive CPU convolution
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += I[((n * C + c) * H + h) * W + w] *
                       F[((k * C + c) * R + r) * S + s];
              }
            }
          }
          O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }
}

__global__ void convolution_gpu(float *I, float *F, float *O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  int ow = blockDim.x * blockIdx.x + threadIdx.x;
  int oh = blockDim.y * blockIdx.y + threadIdx.y;
  if(ow >= OW || oh >=OH) return;
  for (int oc = 0; oc < OC; ++oc) {
    float sum = 0;
    for (int c = 0; c < C; ++c) {
      for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
          const int h = oh * stride_h - pad_h + r * dilation_h;
          const int w = ow * stride_w - pad_w + s * dilation_w;
          const int k = oc;
          if (h < 0 || h >= H || w < 0 || w >= W) continue;
          sum += I[(c * H + h) * W + w] *
                  F[((k * C + c) * R + r) * S + s];
        }
      }
    }
    O[(oc * OH + oh) * OW + ow] = sum;
  }
}

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w) {
  // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution(_I, _F, _O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
  //                       stride_w, dilation_h, dilation_w);

  int Ibegin[10],Iend[10];
  int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  for(int i=0; i<N; i++){
      Ibegin[i] = C*H*W * i;
      Iend[i+1] = C*H*W * (i+1);
  }
  CHECK_CUDA(cudaMemcpyAsync(F_gpu, _F, K*C*R*S*sizeof(float), cudaMemcpyHostToDevice, data_s));
  for(int i=0; i<N; i++){
    CHECK_CUDA(cudaMemcpyAsync(&I_gpu[Ibegin[i]], &_I[Ibegin[i]], C*H*W*sizeof(float), cudaMemcpyHostToDevice, data_s));
    cudaEventRecord(events[i], data_s);
  }

  for(int i=0; i<N; i++){
    dim3 gridDim((OW+31)/32,(OH+31)/32);
    dim3 blockDim(32,32);
    cudaStreamWaitEvent(calc_s, events[i]);
    convolution_gpu<<<gridDim, blockDim, 0, calc_s>>>(&I_gpu[Ibegin[i]], F_gpu, &O_gpu[i*K*OH*OW], N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    cudaGetLastError();
  }
  
  CHECK_CUDA(cudaStreamSynchronize(calc_s));
  CHECK_CUDA(cudaMemcpyAsync(_O, O_gpu, N*K*OH*OW*sizeof(float),cudaMemcpyDeviceToHost,data_s));

  CHECK_CUDA(cudaStreamSynchronize(data_s));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  
  int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  cudaStreamCreate(&data_s);
  cudaStreamCreate(&calc_s);
  for(int i=0; i<N; i++){
    cudaEventCreate(&events[i]);
  }
  
  CHECK_CUDA(cudaMalloc(&I_gpu, N*C*H*W*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F_gpu, K*C*R*S*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O_gpu, N*K*OH*OW*sizeof(float)));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}