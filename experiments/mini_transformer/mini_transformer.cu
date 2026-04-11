#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// Mini Transformer Inference - Pure CUDA (no PyTorch/cuBLAS)
//
// Simulates one decode step of a simplified Transformer.
// Not numerically correct - designed to produce realistic
// GPU execution patterns for trace-driven analysis.
// ============================================================

// --- Configuration ---
#define HIDDEN_DIM   768
#define NUM_HEADS    12
#define HEAD_DIM     64
#define SEQ_LEN      512
#define FFN_DIM      3072
#define NUM_LAYERS   6
#define TILE_SIZE    16

// --- Utility ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernel 1: Tiled GEMM (compute-bound)
// C[M x N] = A[M x K] x B[K x N]
// ============================================================
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================
// Kernel 2: Attention Score (Q x K^T per head)
// Shared memory tiling: each block preloads a HEAD_DIM-wide
// slice of Q (one row per threadIdx.y) and K (one row per
// threadIdx.x) into shared memory, eliminating repeated LDG
// latency stalls from global memory (Class B-4 fix).
//
// Block layout: (TILE_SIZE x TILE_SIZE) = (16 x 16).
// Each block computes a TILE_SIZE x TILE_SIZE tile of scores.
// Q tile: TILE_SIZE rows x HEAD_DIM cols  (threadIdx.y selects row)
// K tile: TILE_SIZE rows x HEAD_DIM cols  (threadIdx.x selects row)
// HEAD_DIM=64 fits in shared memory per block:
//   2 * 16 * 64 * 4B = 8192B < 48KB limit.
// ============================================================
__global__ void attention_score(const float* Q, const float* K,
                                float* scores,
                                int seq_len, int head_dim, int num_heads) {
    __shared__ float Qs[TILE_SIZE][HEAD_DIM];
    // +1 padding eliminates bank conflicts: stride HEAD_DIM=64 is a multiple
    // of 32 banks, so Ks[x][d] and Ks[x+1][d] would collide without padding.
    __shared__ float Ks[TILE_SIZE][HEAD_DIM + 1];

    int head = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Cooperatively load Q tile: each thread loads head_dim/TILE_SIZE elements
    // for its assigned row (threadIdx.y), cycling across columns via threadIdx.x.
    if (row < seq_len && head < num_heads) {
        int q_base = head * seq_len * head_dim + row * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += TILE_SIZE)
            Qs[threadIdx.y][d] = Q[q_base + d];
    }

    // Cooperatively load K tile: threadIdx.y selects which K row.
    int k_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (k_row < seq_len && head < num_heads) {
        int k_base = head * seq_len * head_dim + k_row * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += TILE_SIZE)
            Ks[threadIdx.y][d] = K[k_base + d];
    }

    __syncthreads();

    if (row < seq_len && col < seq_len && head < num_heads) {
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            sum += Qs[threadIdx.y][d] * Ks[threadIdx.x][d];

        scores[head * seq_len * seq_len + row * seq_len + col] =
            sum / sqrtf((float)head_dim);
    }
}

// ============================================================
// Kernel 3: Softmax (per-row, parallel reduction per block)
// One block per row; threads cooperate via shared memory
// reduction. Fixes the 24-block launch underutilization:
// total_rows = NUM_HEADS * SEQ_LEN = 6144 blocks.
// ============================================================
__global__ void softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    float* row_data = data + row * cols;

    // parallel max
    float local_max = -1e38f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_data[j]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // parallel exp + sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        row_data[j] = expf(row_data[j] - max_val);
        local_sum += row_data[j];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float total_sum = sdata[0];

    // normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_data[j] /= total_sum;
}

// ============================================================
// Kernel 4: Context multiply (scores x V per head)
// ============================================================
__global__ void context_mul(const float* scores, const float* V,
                            float* context,
                            int seq_len, int head_dim, int num_heads) {
    int head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < head_dim && head < num_heads) {
        float sum = 0.0f;
        int s_offset = head * seq_len * seq_len + row * seq_len;
        int v_offset = head * seq_len * head_dim;
        for (int j = 0; j < seq_len; j++) {
            sum += scores[s_offset + j] * V[v_offset + j * head_dim + col];
        }
        context[head * seq_len * head_dim + row * head_dim + col] = sum;
    }
}

// ============================================================
// Kernel 5: Layer Normalization (reduction + normalize)
// ============================================================
__global__ void layernorm_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    float* row_data = x + row * cols;

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_sum += row_data[j];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / cols;

    float local_var = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float diff = row_data[j] - mean;
        local_var += diff * diff;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float variance = sdata[0] / cols;
    float inv_std = rsqrtf(variance + 1e-5f);

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_data[j] = (row_data[j] - mean) * inv_std;
}

// ============================================================
// Kernel 6: Residual Add (pure memory-bound)
// ============================================================
__global__ void residual_add(float* output, const float* residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] += residual[idx];
}

// ============================================================
// Helper: Launch GEMM
// ============================================================
void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("Mini Transformer Inference\n");
    printf("  hidden_dim=%d, num_heads=%d, head_dim=%d\n",
           HIDDEN_DIM, NUM_HEADS, HEAD_DIM);
    printf("  seq_len=%d, ffn_dim=%d, num_layers=%d\n",
           SEQ_LEN, FFN_DIM, NUM_LAYERS);

    // --- Allocate device memory ---
    size_t input_size = SEQ_LEN * HIDDEN_DIM * sizeof(float);
    size_t qkv_weight_size = HIDDEN_DIM * HIDDEN_DIM * sizeof(float);
    size_t ffn1_weight_size = HIDDEN_DIM * FFN_DIM * sizeof(float);
    size_t ffn2_weight_size = FFN_DIM * HIDDEN_DIM * sizeof(float);
    size_t scores_size = NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float);
    size_t qkv_size = SEQ_LEN * HIDDEN_DIM * sizeof(float);
    size_t ffn_mid_size = SEQ_LEN * FFN_DIM * sizeof(float);

    float *d_input, *d_residual;
    float *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Wffn1, *d_Wffn2;
    float *d_Q, *d_K, *d_V, *d_scores, *d_context;
    float *d_attn_out, *d_ffn_mid, *d_ffn_out;

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_residual, input_size));
    CUDA_CHECK(cudaMalloc(&d_Wq, qkv_weight_size));
    CUDA_CHECK(cudaMalloc(&d_Wk, qkv_weight_size));
    CUDA_CHECK(cudaMalloc(&d_Wv, qkv_weight_size));
    CUDA_CHECK(cudaMalloc(&d_Wo, qkv_weight_size));
    CUDA_CHECK(cudaMalloc(&d_Wffn1, ffn1_weight_size));
    CUDA_CHECK(cudaMalloc(&d_Wffn2, ffn2_weight_size));
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_context, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_attn_out, input_size));
    CUDA_CHECK(cudaMalloc(&d_ffn_mid, ffn_mid_size));
    CUDA_CHECK(cudaMalloc(&d_ffn_out, input_size));

    // Random init
    {
        size_t max_size = ffn1_weight_size;
        float* h_buf = (float*)malloc(max_size);
        srand(42);
        for (size_t i = 0; i < max_size / sizeof(float); i++)
            h_buf[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        CUDA_CHECK(cudaMemcpy(d_input, h_buf, input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Wq, h_buf, qkv_weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Wk, h_buf, qkv_weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Wv, h_buf, qkv_weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Wo, h_buf, qkv_weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Wffn1, h_buf, ffn1_weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Wffn2, h_buf, ffn2_weight_size, cudaMemcpyHostToDevice));
        free(h_buf);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Memory allocated and initialized.\n");

    // --- Warmup ---
    launch_gemm(d_input, d_Wq, d_Q, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Profiled region ---
    cudaProfilerStart();

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        // Save input for residual
        CUDA_CHECK(cudaMemcpy(d_residual, d_input, input_size,
                              cudaMemcpyDeviceToDevice));

        // 1. QKV projections (3 GEMMs)
        launch_gemm(d_input, d_Wq, d_Q, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM);
        launch_gemm(d_input, d_Wk, d_K, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM);
        launch_gemm(d_input, d_Wv, d_V, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM);

        // 2. Attention scores
        {
            dim3 block(16, 16);
            dim3 grid((SEQ_LEN + 15) / 16, (SEQ_LEN + 15) / 16, NUM_HEADS);
            attention_score<<<grid, block>>>(d_Q, d_K, d_scores,
                                             SEQ_LEN, HEAD_DIM, NUM_HEADS);
        }

        // 3. Softmax — one block per row, 256 threads per block
        {
            int total_rows = NUM_HEADS * SEQ_LEN;
            softmax_kernel<<<total_rows, 256, 256 * sizeof(float)>>>(
                d_scores, total_rows, SEQ_LEN);
        }

        // 4. Context = scores x V
        {
            dim3 block(16, 16);
            dim3 grid((HEAD_DIM + 15) / 16, (SEQ_LEN + 15) / 16, NUM_HEADS);
            context_mul<<<grid, block>>>(d_scores, d_V, d_context,
                                         SEQ_LEN, HEAD_DIM, NUM_HEADS);
        }

        // 5. Output projection
        launch_gemm(d_context, d_Wo, d_attn_out, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM);

        // 6. Residual add
        {
            int n = SEQ_LEN * HIDDEN_DIM;
            int bs = 256;
            residual_add<<<(n + bs - 1) / bs, bs>>>(d_attn_out, d_residual, n);
        }

        // 7. Layer norm
        layernorm_kernel<<<SEQ_LEN, 256, 256 * sizeof(float)>>>(
            d_attn_out, SEQ_LEN, HIDDEN_DIM);

        // Save for FFN residual
        CUDA_CHECK(cudaMemcpy(d_residual, d_attn_out, input_size,
                              cudaMemcpyDeviceToDevice));

        // 8. FFN layer 1 (768 -> 3072)
        launch_gemm(d_attn_out, d_Wffn1, d_ffn_mid, SEQ_LEN, FFN_DIM, HIDDEN_DIM);

        // 9. FFN layer 2 (3072 -> 768)
        launch_gemm(d_ffn_mid, d_Wffn2, d_ffn_out, SEQ_LEN, HIDDEN_DIM, FFN_DIM);

        // 10. Residual add
        {
            int n = SEQ_LEN * HIDDEN_DIM;
            int bs = 256;
            residual_add<<<(n + bs - 1) / bs, bs>>>(d_ffn_out, d_residual, n);
        }

        // 11. Layer norm
        layernorm_kernel<<<SEQ_LEN, 256, 256 * sizeof(float)>>>(
            d_ffn_out, SEQ_LEN, HIDDEN_DIM);

        // Next layer input
        CUDA_CHECK(cudaMemcpy(d_input, d_ffn_out, input_size,
                              cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();

    printf("Inference complete (%d layers).\n", NUM_LAYERS);

    // Cleanup
    cudaFree(d_input); cudaFree(d_residual);
    cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_Wo);
    cudaFree(d_Wffn1); cudaFree(d_Wffn2);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_scores); cudaFree(d_context);
    cudaFree(d_attn_out); cudaFree(d_ffn_mid); cudaFree(d_ffn_out);

    printf("Done.\n");
    return 0;
}
