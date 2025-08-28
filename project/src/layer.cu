#include "layer.h"
#include <omp.h>

/*
  * PixelNorm
  * @param [in & out] inout: [N, C]
  * Normalizes the input tensor along dim=1.
  * Equivalent to: input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
  */

void PixelNorm(Tensor *inout) {
  size_t N = inout->shape[0];
  size_t C = inout->shape[1];

  float mean_squares, norm_factor;
  
  for (size_t n = 0; n < N; n++) {
    mean_squares = 0.f;
    for (size_t i = 0; i < C; i++) {
      mean_squares += inout->buf[n * C + i] * inout->buf[n * C + i];
    }
    mean_squares /= C;
    norm_factor = rsqrtf(mean_squares + 1e-8f);

    for (size_t i = 0; i < C; i++) {
      inout->buf[n * C + i] *= norm_factor;
    }
  }
}

__global__ void pixelNormKernel(float *inout, size_t N, size_t C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float *data = inout + idx * C;
    
    float mean_squares = 0.0f;
    for (size_t i = 0; i < C; i++) {
        mean_squares += data[i] * data[i];
    } 
    mean_squares /= C;
    
    float norm_factor = rsqrtf(mean_squares + 1e-8f);
    for (size_t i = 0; i < C; i++) {
        data[i] *= norm_factor;
    }
}

void PixelNorm_gpu(Tensor *inout) {
    size_t N = inout->shape[0];
    size_t C = inout->shape[1];
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    pixelNormKernel<<<grid, block>>>(inout->gpu_buf, N, C);
    CHECK_CUDA(cudaGetLastError());
}

/*
 * Upsample and Pad
 * input shape = (N, C, H, W)
 * output shape = (N, C, OH, OW)
 *   where OH = H * up + pad0 + pad1,
 *         OW = W * up + pad0 + pad1
 */
void UpsamplePad(Tensor *input, Tensor *output, int up, int pad0, int pad1) {
  size_t N = input->shape[0];
  size_t C = input->shape[1]; 
  size_t H = input->shape[2];
  size_t W = input->shape[3];

  size_t OH = up * H + pad0 + pad1;
  size_t OW = up * W + pad0 + pad1;

  memset(output->buf, 0, N * C * OH * OW * sizeof(float));
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                output->buf[n * C * OH * OW + c * OH * OW + (h * up + pad0) * OW + w * up + pad0] +=
                    input->buf[n * C * H * W + c * H * W + h * W + w];
            }
        }
    }
  }
}

__global__ void UpsamplePadKernel(Tensor *input, Tensor *output, int up, int pad0, int pad1) {
  size_t N = input->shape[0];
  size_t C = input->shape[1]; 
  size_t H = input->shape[2];
  size_t W = input->shape[3];

  size_t OH = up * H + pad0 + pad1;
  size_t OW = up * W + pad0 + pad1;

  memset(output->buf, 0, N * C * OH * OW * sizeof(float));
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                output->buf[n * C * OH * OW + c * OH * OW + (h * up + pad0) * OW + w * up + pad0] +=
                    input->buf[n * C * H * W + c * H * W + h * W + w];
            }
        }
    }
  }
}

void UpsamplePad_gpu(Tensor *input, Tensor *output, int up, int pad0, int pad1) {
  dim3 gridDim;
}

/*
 * Convolution
 * input shape = (N, C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */
void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias) {
  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[0], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float o = has_bias ? bias->buf[k] : 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                int h = oh * stride - pad + r * dilation;
                int w = ow * stride - pad + s * dilation;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                float i = input->buf[n * C * H * W + c * H * W + h * W + w];
                float f = weight->buf[k * C * R * S + c * R * S + r * S + s];
                o += i * f;
              }
            }
          }
          output->buf[n * K * OH * OW + k * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
}

/*
 * Grouped Convolution
 * input shape = (N, C, H, W)
 * weight shape = (N, K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */
void GroupedConv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias) {
  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[1], R = weight->shape[3], S = weight->shape[4];
  int OH = output->shape[2], OW = output->shape[3];

  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float o = has_bias ? bias->buf[k] : 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                int h = oh * stride - pad + r * dilation;
                int w = ow * stride - pad + s * dilation;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                float i = input->buf[n * C * H * W + c * H * W + h * W + w];
                float f = weight->buf[n * K * C * R * S + k * C * R * S + c * R * S + r * S + s];
                o += i * f;
              }
            }
          }
          output->buf[n * K * OH * OW + k * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
}


/*
 * Grouped Transposed convolution
 * input shape = (N, C, H, W)
 * weight shape = (N, C, K, R, S)
 * output shape = (N, K, OH, OW)
 *   where OH = (H - 1) * stride - 2 * pad + R
 *         OW = (W - 1) * stride - 2 * pad + S
 */
void GroupedConvTranspose2d(Tensor *input, Tensor *weight, Tensor *output, 
                     int stride, int pad) {
  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[2], R = weight->shape[3], S = weight->shape[4];
  int OH = output->shape[2], OW = output->shape[3];

  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float o = 0.0f;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                if ((oh + pad - r) % stride != 0) continue;
                if ((ow + pad - s) % stride != 0) continue;
                int h = (oh + pad - r) / stride;
                int w = (ow + pad - s) / stride;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                float i = input->buf[n * C * H * W + c * H * W + h * W + w];
                float f = weight->buf[n * C * K * R * S + c * K * R * S + k * R * S + r * S + s];
                o += i * f;
              }
            }
          }
          output->buf[n * K * OH * OW + k * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
}

/* Transpose
 * input shape = (N, out_C, in_C, H, W)
 * output shape = (N, in_C, out_C, H, W)
 * Transposes the first two dimensions of the input tensor.
 */
void transpose(Tensor *input, Tensor *output) {
  size_t N = input->shape[0];
  size_t out_C = input->shape[1];
  size_t in_C = input->shape[2];
  size_t H = input->shape[3];
  size_t W = input->shape[4];

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < out_C; oc++) {
      for (size_t ic = 0; ic < in_C; ic++) {
        for (size_t h = 0; h < H; h++) {
          for (size_t w = 0; w < W; w++) {
            size_t input_idx  = (((n * out_C + oc) * in_C + ic) * H + h) * W + w;
            size_t output_idx = (((n * in_C + ic) * out_C + oc) * H + h) * W + w;
            output->buf[output_idx] = input->buf[input_idx];
          }
        }
      }
    }
  }
}

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  float scale = (1.0f / sqrtf(K)) * lr_mul;

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[m * N + n] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k] * scale;
      }
      out->buf[m * N + n] += b->buf[n] * lr_mul;
    }
  }
}

__global__ void LinearKernel(float *in, float *w, float *b, float *out, float lr_mul, size_t M, size_t N, size_t K) {
  float scale = (1.0f / sqrtf(K)) * lr_mul;
  size_t m = blockDim.x * blockIdx.x + threadIdx.x;
  size_t n = blockDim.y * blockIdx.y + threadIdx.y;
  if(m >= M || n >= N) return;

  out[m * N + n] = 0;
  for (size_t k = 0; k < K; k++) {
    out[m * N + n] += in[m * K + k] * w[n * K + k] * scale;
  }
  out[m * N + n] += b[n] * lr_mul;
}

void Linear_gpu(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];
  w->to_gpu();
  b->to_gpu();

  dim3 gridDim((M+31)/32,(N+31)/32);
  dim3 blockDim(32,32);

  LinearKernel<<<gridDim, blockDim>>>(in->gpu_buf, w->gpu_buf, b->gpu_buf, out->gpu_buf, lr_mul, M, N, K);
  CHECK_CUDA(cudaGetLastError());
}

/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  float negative_slope = 0.2f;
  float scale = sqrtf(2.0f);

  for (size_t i = 0; i < N; i++) {
    if (inout->buf[i] < 0) { inout->buf[i] *= negative_slope; }
    inout->buf[i] *= scale;
  }
}

__global__ void LeakyReLUKernel(float *inout, size_t N, float scale) {

  float negative_slope = 0.2f;

  for (size_t i = 0; i < N; i++) {
    if (inout[i] < 0) { inout[i] *= negative_slope; }
    inout[i] *= scale;
  }
}

void LeakyReLU_gpu(Tensor *inout) {
  size_t N = inout->num_elem();
  float scale = sqrtf(2.0f);

  dim3 gridDim(1);
  dim3 blockDim(1);

  LeakyReLUKernel<<<gridDim, blockDim>>>(inout->gpu_buf, N, scale);
  CHECK_CUDA(cudaGetLastError());
}

void upfir2d(Tensor *input, Tensor *kernel, Tensor *output,
               Tensor *upsample_a, Tensor *conv_a,
               int up, int pad0, int pad1) {
  // Upsample and Pad -> Conv2d (FIR filter)
  UpsamplePad(input, upsample_a, up, pad0, pad1);

  int N = upsample_a->shape[0];
  int C = upsample_a->shape[1];
  int H = upsample_a->shape[2];
  int W = upsample_a->shape[3];
  upsample_a->reshape({N * C, 1, H, W});
  Conv2d(upsample_a, kernel, nullptr, output, 1, 0, 1, false);
  upsample_a->reshape({N, C, H, W});
}

/*
 * Modulate Weights
 * Applies style modulation to convolution weights.
 * @param [in] conv_weight: Original convolution weights [out_C, in_C, kernel_size, kernel_size]
 * @param [in] style_a: Style vector [N, in_C]
 * @param [out] weight_a: Modulated weights [N, out_C, in_C, kernel_size, kernel_size]
 * @param [in] scale: Scaling factor
 */
void ModulateWeights(Tensor *conv_weight, Tensor *style_a, Tensor *weight_a, float scale) {
  size_t N = style_a->shape[0];
  size_t out_C = conv_weight->shape[0];
  size_t in_C = conv_weight->shape[1];
  size_t kernel_size = conv_weight->shape[2];

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < out_C; oc++) {
      for (size_t ic = 0; ic < in_C; ic++) {
        for (size_t k = 0; k < kernel_size * kernel_size; k++) {
          size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
          weight_a->buf[n * out_C * in_C * kernel_size * kernel_size + idx] = conv_weight->buf[idx] * style_a->buf[n * in_C + ic] * scale;
        }
      }
    }
  }
}

/*
 * Compute Demodulation Factors
 * Computes demodulation factors based on the modulated weights.
 * @param [in] weight_a: Modulated weights [N, out_C, in_C, kernel_size, kernel_size]
 * @param [out] demod_a: Demodulation factors [N, out_C]
 */
void ComputeDemodulationFactors(Tensor *weight_a, Tensor *demod_a) {
  size_t N = weight_a->shape[0];
  size_t out_C = weight_a->shape[1];
  size_t in_C = weight_a->shape[2];
  size_t kernel_size = weight_a->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < out_C; oc++) {
      float sum = 0.0f;
      for (size_t ic = 0; ic < in_C; ic++) {
        for (size_t k = 0; k < kernel_size * kernel_size; k++) {
          size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
          sum += weight_a->buf[n * out_C * in_C * kernel_size * kernel_size + idx] * weight_a->buf[n * out_C * in_C * kernel_size * kernel_size + idx];
        }
      }
      demod_a->buf[n * out_C + oc] = 1.0f / sqrtf(sum + 1e-8f);
    }
  }
}

/*
 * Apply Demodulation
 * Applies demodulation factors to the modulated weights.
 * @param [in & out] weight_a: Modulated weights [N, out_C, in_C, kernel_size, kernel_size]
 * @param [in] demod_a: Demodulation factors [N, out_C]
 */
void ApplyDemodulation(Tensor *weight_a, Tensor *demod_a) {
  size_t N = weight_a->shape[0];
  size_t out_C = weight_a->shape[1];
  size_t in_C = weight_a->shape[2];
  size_t kernel_size = weight_a->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < out_C; oc++) {
      for (size_t ic = 0; ic < in_C; ic++) {
        for (size_t k = 0; k < kernel_size * kernel_size; k++) {
          size_t idx = n * out_C * in_C * kernel_size * kernel_size + oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
          weight_a->buf[idx] *= demod_a->buf[n * out_C + oc];
        }
      }
    }
  }
}

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, int padding, int up
) {
  int in_C = input->shape[1];
  int kernel_size = conv_weight->shape[2];

  Linear(style, modulate_weight, modulate_bias, style_a, 1.0f);

  float scale = 1 / sqrtf((float) (in_C * kernel_size * kernel_size));

  ModulateWeights(conv_weight, style_a, weight_a, scale);

  if (demodulate) {
    ComputeDemodulationFactors(weight_a, demod_a);
    ApplyDemodulation(weight_a, demod_a);
  }

  if (upsample) {
    transpose(weight_a, transpose_a);
    GroupedConvTranspose2d(input, transpose_a, conv_a, 2, 0);
    upfir2d(conv_a, kernel, output, upsample_a, conv2_a, up, 1, 1);
  }
  else {
    GroupedConv2d(input, weight_a, nullptr, output, 1, padding, 1, false);
  }
}

/* Add noise to the input tensor
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] noise: [H, W]
 * Adds noise to the input tensor in-place.
 */

void addNoise(Tensor *inout, Tensor *noise) {
  size_t N = inout->shape[0];
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t idx = (((n * C + c) * H + h) * W + w);
          inout->buf[idx] += noise->buf[h * W + w];
        }
      }
    }
  }
}

/* Add bias to the input tensor
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] bias: [C]
 * Adds bias to the input tensor in-place.
 */

void addBias(Tensor *inout, Tensor *bias) {
  size_t N = inout->shape[0];
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t idx = (((n * C + c) * H + h) * W + w);
          inout->buf[idx] += bias->buf[c];
        }
      }
    }
  }
}

/*
 * Element-wise addition of two tensors
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] addend: [N, C, H, W]
 * Adds the elements of addend to inout in-place.
 */
void elemAdd(Tensor *inout, Tensor *addend) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] += addend->buf[i];
  }
}

void StyledConv(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *noise, Tensor *output,
                Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                bool demodulate, bool upsample, int padding) {
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, demod_a, transpose_a, conv_a, upsample_a, conv2_a,
                  demodulate, upsample, padding, 1);
  addNoise(output, noise);
  addBias(output, conv_bias);
  LeakyReLU(output);
}

void ToRGB(Tensor *input, Tensor *skip, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *output,
           Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a,
           bool demodulate, bool upsample, int padding) {
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, demod_a, transpose_a, conv_a, upsample_a, conv2_a,
                  demodulate, upsample, padding, 2);
  addBias(output, conv_bias);

  if (skip != nullptr) {
    upfir2d(skip, kernel, skip_a, skip_upsample_a, skip_conv_a, 2, 2, 1);
    elemAdd(output, skip_a);
  }
}