// Inference-only compact scalar heads. One block per sample; no Torch ABI dependency.
#include <cuda_runtime.h>
#include <math.h>

__device__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

__device__ float affine(const float* weights, const float* z, int n, float bias) {
    float value = bias;
    for (int j = 0; j < n; ++j) value = fmaf(weights[j], z[j], value);
    return value;
}

__device__ float finite_number(float x, float nan, float positive, float negative) {
    return isnan(x) ? nan : (isinf(x) ? (x > 0 ? positive : negative) : x);
}

__device__ float eml(float a, float b) {
    a = fminf(80.0f, fmaxf(-80.0f, finite_number(a, 0.0f, 80.0f, -80.0f)));
    b = fminf(1e30f, fmaxf(1e-6f, finite_number(b, 1.0f, 1e30f, -1e30f)));
    return finite_number(expf(a) - logf(b), 0.0f, 1e30f, -1e30f);
}

__global__ void scalar_head(
    const float* input, const float* projection, const float* projection_bias,
    const float* skip, const float* left, const float* left_bias,
    const float* right, const float* right_bias, const float* second,
    const float* second_bias, const float* output_weights, float* output,
    int dimension, int rank, int width, int kind, float output_mean, float output_scale
) {
    __shared__ float z[64];
    __shared__ float hidden[64];
    __shared__ float transformed[64];
    const int lane = threadIdx.x % 32;
    const int feature = threadIdx.x / 32;
    const float* x = input + blockIdx.x * dimension;
    if (feature < rank) {
        if (projection) {
            float value = 0.0f;
            for (int j = lane; j < dimension; j += 32)
                value = fmaf(x[j], projection[feature * dimension + j], value);
            value = warp_sum(value);
            if (lane == 0) z[feature] = value + projection_bias[feature];
        } else if (lane == 0) z[feature] = x[feature];
    }
    __syncthreads();
    int i = threadIdx.x;
    if (i < width) {
        float a = affine(left + i * rank, z, rank, left_bias[i]);
        float b = affine(right + i * rank, z, rank, right_bias[i]);
        if (kind == 0) hidden[i] = eml(a, fmaf(b, b, 1.0f));
        else if (kind == 1) hidden[i] = eml(0.0f, fmaf(b, b, 1.0f));
        else if (kind == 2) hidden[i] = eml(a, b);
        else hidden[i] = a / (1.0f + expf(-a));
    }
    __syncthreads();
    if (i < width) {
        if (kind == 4) {
            float a = affine(second + i * width, hidden, width, second_bias[i]);
            transformed[i] = a / (1.0f + expf(-a));
        } else transformed[i] = hidden[i];
    }
    __syncthreads();
    if (i == 0) {
        float value = affine(skip, z, rank, skip[rank]);
        for (int j = 0; j < width; ++j) value = fmaf(output_weights[j], transformed[j], value);
        output[blockIdx.x] = fmaf(value, output_scale, output_mean);
    }
}

__global__ void project_features(
    const float* input, const float* projection, const float* bias,
    float* projected, int dimension, int rank
) {
    __shared__ float partial[4];
    const int feature = blockIdx.y;
    const float* x = input + blockIdx.x * dimension;
    const float* w = projection + feature * dimension;
    float value = 0.0f;
    for (int j = threadIdx.x; j < dimension; j += blockDim.x)
        value = fmaf(x[j], w[j], value);
    value = warp_sum(value);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x == 0)
        projected[blockIdx.x * rank + feature] = partial[0] + partial[1] + partial[2] + partial[3] + bias[feature];
}

extern "C" int launch_head_split(
    const float* input, const float* projection, const float* projection_bias,
    const float* skip, const float* left, const float* left_bias,
    const float* right, const float* right_bias, const float* second,
    const float* second_bias, const float* output_weights, float* output, float* projected,
    int batch, int dimension, int rank, int width, int kind,
    float output_mean, float output_scale, cudaStream_t stream
) {
    if (rank < 1 || rank > 32 || width < 1 || width > 64 || batch < 1) return -1;
    project_features<<<dim3(batch, rank), 128, 0, stream>>>(
        input, projection, projection_bias, projected, dimension, rank
    );
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    scalar_head<<<batch, (rank < 2 ? 2 : rank) * 32, 0, stream>>>(
        projected, nullptr, nullptr, skip, left, left_bias,
        right, right_bias, second, second_bias, output_weights, output,
        rank, rank, width, kind, output_mean, output_scale
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_head(
    const float* input, const float* projection, const float* projection_bias,
    const float* skip, const float* left, const float* left_bias,
    const float* right, const float* right_bias, const float* second,
    const float* second_bias, const float* output_weights, float* output,
    int batch, int dimension, int rank, int width, int kind,
    float output_mean, float output_scale, cudaStream_t stream
) {
    if (rank < 1 || rank > 32 || width < 1 || width > 64 || batch < 1) return -1;
    scalar_head<<<batch, (rank < 2 ? 2 : rank) * 32, 0, stream>>>(
        input, projection, projection_bias, skip, left, left_bias,
        right, right_bias, second, second_bias, output_weights, output,
        dimension, rank, width, kind, output_mean, output_scale
    );
    return static_cast<int>(cudaGetLastError());
}
