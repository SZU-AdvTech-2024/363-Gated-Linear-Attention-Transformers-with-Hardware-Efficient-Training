#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"


AUTO_CUDA_KERNEL_PRE_DEFINE
// typedef  ftype;
// typedef  ctype;
// constexpr int siz_head = SIZ_HEAD;  // 32x , = GPU warp size for performance


template <typename dtype>
__global__ void kernel_forward(const int num_token, const int dim_feature,
                               const dtype * __restrict__ const q_global,
                               const dtype * __restrict__ const k_global,
                               const dtype * __restrict__ const g_global,
                               const dtype * __restrict__ const v_global,
                                     dtype * __restrict__ const y_global,
                                     dtype * __restrict__ const s_global)
{
    const int idx_batch   = blockIdx.x;
    const int idx_head    = blockIdx.y;
    const int idx_feature = threadIdx.x;

    __shared__ ctype q[siz_head], k[siz_head], g[siz_head];


    // s (siz_batch, dim_feature=(num_head, siz_head), siz_head)
    const int bgn_s =  idx_batch     * dim_feature            * siz_head +
                       idx_head                    * siz_head * siz_head +
                       0                                      * siz_head +
                       idx_feature;

    // load s0 to reg, align state (dim_k=-2)
    ctype s[siz_head] = {0};
    #pragma unroll
    for(int index = 0; index < siz_head; index++){
        s[index] = ctype(s_global[bgn_s + index * siz_head]);
    }


    // q k v y (siz_batch, num_token, dim_feature=(num_head, siz_head))
    const int bgn_seq_feature =  idx_batch      * num_token * dim_feature            +
                                 0                          * dim_feature            +
                                 idx_head                                 * siz_head +
                                 idx_feature;
    const int end_seq_feature = (idx_batch + 1) * num_token * dim_feature            +
                                 0                          * dim_feature            +
                                 idx_head                                 * siz_head +
                                 idx_feature;


    // from token0 to tokenN , calculate y
    for (int idx_token_feature = bgn_seq_feature;
             idx_token_feature < end_seq_feature;
             idx_token_feature +=    dim_feature){

        // load global data to shared mem and reg
        __syncthreads();
        q[idx_feature]    = ctype(q_global[idx_token_feature]);
        k[idx_feature]    = ctype(k_global[idx_token_feature]);
        g[idx_feature]    = ctype(g_global[idx_token_feature]);
        const ctype v_val = ctype(v_global[idx_token_feature]);
        __syncthreads();


        ctype y_val = 0;

        // calculate y_t
        #pragma unroll
        for(int index = 0; index < siz_head; index++){

            s[index] = g[index] * s[index] + k[index] * v_val;
            y_val += q[index] * s[index];
        }


        //return y_token_feature
        y_global[idx_token_feature] = dtype(y_val);
    }


    //return s_final
    #pragma unroll
    for(int index = 0; index < siz_head; index++){
         s_global[bgn_s + index * siz_head] = dtype(s[index]);
    }
}


template <typename dtype>
__global__ void kernel_backward(const int num_token, const int dim_feature,
                                const dtype * __restrict__ const q_global,
                                const dtype * __restrict__ const k_global,
                                const dtype * __restrict__ const g_global,
                                const dtype * __restrict__ const v_global,
                                const dtype * __restrict__ const s_global,
                                const dtype * __restrict__ const grad_y_global,
                                      dtype * __restrict__ const grad_s_global,
                                      dtype * __restrict__ const grad_q_global,
                                      dtype * __restrict__ const grad_k_global,
                                      dtype * __restrict__ const grad_g_global,
                                      dtype * __restrict__ const grad_v_global,
                                      ctype * __restrict__ const cache_grad_gkv_global,
                                      ctype * __restrict__ const cache_cumprod_g_global)
{
    const int idx_batch   = blockIdx.x;
    const int idx_head    = blockIdx.y;
    const int idx_feature = threadIdx.x;

    __shared__ ctype q[siz_head], k[siz_head], g[siz_head], v[siz_head], grad_y[siz_head];


    // s (siz_batch, dim_feature=(num_head, siz_head), siz_head)
    const int bgn_s =  idx_batch     * dim_feature            * siz_head +
                       idx_head                    * siz_head * siz_head +
                       0                                      * siz_head +
                       0;

    // load grad_s to reg, align both state(dim_k=-2) and feature(dim_v=-1)
    ctype grad_s_align_k[siz_head] = {0}, grad_s_align_v[siz_head] = {0};
    #pragma unroll
    for(int index = 0; index < siz_head; index++){
        grad_s_align_k[index] = ctype(grad_s_global[bgn_s + index       * siz_head + idx_feature]);
        grad_s_align_v[index] = ctype(grad_s_global[bgn_s + idx_feature * siz_head + index      ]);
    }


    // q k v y (siz_batch, num_token, dim_feature=(num_head, siz_head))
    const int bgn_seq_feature =  idx_batch      * num_token * dim_feature            +
                                 0                          * dim_feature            +
                                 idx_head                                 * siz_head +
                                 idx_feature;
    const int end_seq_feature = (idx_batch + 1) * num_token * dim_feature            +
                                 0                          * dim_feature            +
                                 idx_head                                 * siz_head +
                                 idx_feature;


    ctype cache_yq_align_k[siz_head] = {0}, cache_yq_align_v[siz_head] = {0};
    ctype cumprod_g_align_k[siz_head] = {0}, cumprod_g_align_v_val = 1;

    for (int index = 0; index < siz_head; index++){
        cumprod_g_align_k[index] = 1;
    }

    // from tokenN to token0 , calculate grad k v and g cache
    for (int idx_token_feature =  end_seq_feature - dim_feature   ;
             idx_token_feature >= bgn_seq_feature                 ;
             idx_token_feature -=     dim_feature                 ){


        // load global data to shared mem and reg
        __syncthreads();
             q[idx_feature]    = ctype(     q_global[idx_token_feature]);
             k[idx_feature]    = ctype(     k_global[idx_token_feature]);
             g[idx_feature]    = ctype(     g_global[idx_token_feature]);
             v[idx_feature]    = ctype(     v_global[idx_token_feature]);
        grad_y[idx_feature]    = ctype(grad_y_global[idx_token_feature]);
        const ctype      q_val = ctype(     q_global[idx_token_feature]);
        const ctype      k_val = ctype(     k_global[idx_token_feature]);
        const ctype      g_val = ctype(     g_global[idx_token_feature]);
        const ctype grad_y_val = ctype(grad_y_global[idx_token_feature]);
        __syncthreads();


        ctype grad_k_val = 0, grad_v_val = 0, grad_g_val = 0;

        // calculate grad k v, and cache the data needed by grad g
        #pragma unroll
        for (int index = 0; index < siz_head; index++){

            ctype yq_align_k = cache_yq_align_k[index] + grad_y_val    * q[index];
            ctype yq_align_v = cache_yq_align_v[index] + grad_y[index] * q_val   ;

            grad_k_val += yq_align_v * v[index];
            grad_v_val += yq_align_k * k[index];

            grad_k_val += grad_s_align_v[index] * cumprod_g_align_v_val    * v[index];
            grad_v_val += grad_s_align_k[index] * cumprod_g_align_k[index] * k[index];

            grad_g_val += cache_yq_align_v[index] * v[index];

            cache_yq_align_k[index] = yq_align_k * g[index];
            cache_yq_align_v[index] = yq_align_v * g_val   ;

            cumprod_g_align_k[index] *= g[index];
        }
        cumprod_g_align_v_val *= g_val;


        // return grad_k_token_feature,  grad_v_token_feature,
        grad_k_global[idx_token_feature] = dtype(grad_k_val);
        grad_v_global[idx_token_feature] = dtype(grad_v_val);

        // cache gkv val, cache cumprod g
        cache_grad_gkv_global[idx_token_feature]  = grad_g_val * k_val;
        cache_cumprod_g_global[idx_token_feature] = cumprod_g_align_v_val;
    }


    // return grad_s
    #pragma unroll
    for(int index = 0; index < siz_head; index++){

        const ctype grad_s_val = grad_s_align_v[index] * cumprod_g_align_v_val + cache_yq_align_v[index];

        grad_s_global[bgn_s + idx_feature * siz_head + index] = dtype(grad_s_val);
    }


    // load s to reg, align feature (dim[-2]),
    ctype s[siz_head] = {0};
    for(int index = 0; index < siz_head; index++){
        s[index] = ctype(s_global[bgn_s + idx_feature * siz_head + index]);
    }


    ctype grad_y_dg_val = 0;

    // calculate grad g0 on y shifted, cuz kvN needed by gN+1 and qN
    #pragma unroll
    for (int index = 0; index < siz_head; index++){
        grad_y_dg_val += cache_yq_align_v[index] * s[index];
    }


    // from token0 to tokenN , calculate grad q g
    for (int idx_token_feature = bgn_seq_feature  ;
             idx_token_feature < end_seq_feature  ;
             idx_token_feature +=    dim_feature  ){

        // load global data to shared mem and reg
        __syncthreads();
        const ctype       q_val   = ctype(     q_global[idx_token_feature]);
        const ctype       k_val   = ctype(     k_global[idx_token_feature]);
        const ctype       g_val   = ctype(     g_global[idx_token_feature]);
             v[idx_feature]       = ctype(     v_global[idx_token_feature]);
        grad_y[idx_feature]       = ctype(grad_y_global[idx_token_feature]);
        const ctype cache_gkv_val = cache_grad_gkv_global[idx_token_feature];  // cached gkv val
        const ctype cumprod_g_val = cache_cumprod_g_global[idx_token_feature];  // cached cumprod g
        __syncthreads();


        ctype grad_q_val = 0, extra_grad_g_val = 0, grad_s_dg_val = 0;

        // calculate grad q g
        #pragma unroll
        for (int index = 0; index < siz_head; index++){

            grad_s_dg_val += grad_s_align_v[index] * cumprod_g_val * s[index];

            extra_grad_g_val += grad_y[index] * s[index];

            s[index] = g_val * s[index] + k_val * v[index];

            grad_q_val += grad_y[index] * s[index];
        }


        // return grad_g_token_feature
        grad_g_global[idx_token_feature] = dtype((grad_y_dg_val + grad_s_dg_val) / (g_val + 1e-5));


        // combine grad gN+1 on grad_y
        grad_y_dg_val = grad_y_dg_val + cache_gkv_val - extra_grad_g_val * q_val * g_val;


        // return grad_q_token_feature
        grad_q_global[idx_token_feature] = dtype(grad_q_val);
    }
}


void cuda_forward(int siz_batch, int num_token, int dim_feature, int num_head,
                  ftype *q, ftype *k, ftype *g, ftype *v,
                  ftype *y, ftype *s){

    dim3 grid_dim(siz_batch, num_head);
    dim3 block_dim(siz_head);

    kernel_forward<<<grid_dim, block_dim>>>(num_token, dim_feature,
                                            q, k, g, v,
                                            y, s);
}


void cuda_backward(int siz_batch, int num_token, int dim_feature, int num_head,
                   ftype *q, ftype *k, ftype *g, ftype *v, ftype *s,
                   ftype *grad_y, ftype *grad_s,
                   ftype *grad_q, ftype *grad_k, ftype *grad_g, ftype *grad_v){

    dim3 grid_dim(siz_batch, num_head);
    dim3 block_dim(siz_head);

    ctype *cache_grad_gkv, *cache_cumprod_g;
    cudaMalloc(&cache_grad_gkv, siz_batch * num_token * dim_feature * sizeof(ctype));
    cudaMalloc(&cache_cumprod_g, siz_batch * num_token * dim_feature * sizeof(ctype));

    kernel_backward<<<grid_dim, block_dim>>>(num_token, dim_feature,
                                             q, k, g, v, s,
                                             grad_y, grad_s,
                                             grad_q, grad_k, grad_g, grad_v,
                                             cache_grad_gkv, cache_cumprod_g);

    cudaFree(cache_grad_gkv);
    cudaFree(cache_cumprod_g);
}


