from jaxtyping import Float, Bool, Int
from torch import Tensor
from einops import rearrange
import torch
import math

import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,

    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr=False
):
    query_tile_index = tl.program_id(0) # Row i, equivalent one iter in i, ..., T
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D), # The size same as Q
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # The reason why is because it is in j loop
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    


    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)

    O_i_orig = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i_orig = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m_i_orig = tl.full((Q_TILE_SIZE, ), -float(torch.inf), dtype=tl.float32)

    O_i_prev = O_i_orig
    l_i_prev = l_i_orig
    m_i_prev = m_i_orig

    # Start the j loop
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            # Get the global location of q/k tile with corresponding value [1, 2, 3..... ]
            q_global = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) # 
            k_global = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

            # q_global = [1, 16], k_global = [16, 1]
            # After broadcasting, q_global [0, 1, ... 15] on each row; k_global [0, 0, ... 0] on row 0, then [1, 1, 1,... 1]
            # k_global > q_global gives diagonal=1 mask
            mask = k_global[None, :] > q_global[:, None] 

            mask_stable = tl.full((Q_TILE_SIZE, K_TILE_SIZE), float(1e-6), dtype=tl.float32)

            S_ij = tl.where(mask, -float("inf"), S_ij)
            S_ij += mask_stable

        m_ij = tl.maximum(m_i_prev,  tl.max(S_ij, axis=-1)) # (Q_TILE_SIZE, )

        P_ij = tl.exp(S_ij - m_ij[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        l_ij = tl.exp(m_i_prev - m_ij) * l_i_prev + tl.sum(P_ij, axis=-1) # (Q_TILE_SIZE, )

        # Casting P_ij to V_j dtype
        P_ij = P_ij.to(V_j.dtype)
        O_ij = tl.exp(m_i_prev - m_ij)[:, None] * O_i_prev + tl.dot(P_ij, V_j) # (Q_TILE_SIZE, D)

        # Moving
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        O_i_prev = O_ij
        l_i_prev = l_ij
        m_i_prev = m_ij


    O_i = (1 / l_i_prev)[:, None] *  O_i_prev
    L_i = m_i_prev + tl.log(l_i_prev)

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0, ))

    Q_block_ptr.advance((Q_TILE_SIZE, 0))
    O_block_ptr.advance((Q_TILE_SIZE, 0))
    L_block_ptr.advance((Q_TILE_SIZE, 0))    

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_k"],
        is_causal=False
        ):

        q_batch, N_QUERIES, D = Q.shape
        k_batch, N_KEYS, D = K.shape

        assert q_batch == k_batch # Make sure batch-dim matches

        device = Q.device

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE= K_TILE_SIZE
        ctx.D = D
        ctx.is_causal = is_causal

        scale = 1/ (D ** 0.5)

        # Make O and L for output
        O = torch.empty(
            size=(q_batch, N_QUERIES, D),
            dtype=torch.float32,
            device=device
        )

        L = torch.empty(
            size=(q_batch, N_QUERIES, ),
            dtype=torch.float32,
            device=device
        )

        if not is_causal:
            flash_fwd_kernel[triton.cdiv(N_QUERIES, Q_TILE_SIZE), q_batch](  # To make grid as 2-D, here need to make q_batch in kernel
                Q, K, V, O, L,

                Q.stride(0), Q.stride(1), Q.stride(2), # stride_qb, stride_qq, stride_qd
                K.stride(0), K.stride(1), K.stride(2), # stride_kb, stride_kk, stride_kd
                V.stride(0), V.stride(1), V.stride(2), # stride_vb, stride_vk, stride_vd
                O.stride(0), O.stride(1), O.stride(2), # stride_ob, stride_oq, stride_od
                L.stride(0), L.stride(1),              # stride_lb, stride_lq, stride_ld  

                N_QUERIES, N_KEYS,
                scale,
                ctx.D,
                ctx.Q_TILE_SIZE,
                ctx.K_TILE_SIZE,
                ctx.is_causal
            )
        else:
            flash_fwd_kernel[triton.cdiv(N_QUERIES, Q_TILE_SIZE), q_batch](  # To make grid as 2-D, here need to make q_batch in kernel
                Q, K, V, O, L,

                Q.stride(0), Q.stride(1), Q.stride(2), # stride_qb, stride_qq, stride_qd
                K.stride(0), K.stride(1), K.stride(2), # stride_kb, stride_kk, stride_kd
                V.stride(0), V.stride(1), V.stride(2), # stride_vb, stride_vk, stride_vd
                O.stride(0), O.stride(1), O.stride(2), # stride_ob, stride_oq, stride_od
                L.stride(0), L.stride(1),              # stride_lb, stride_lq, stride_ld  

                N_QUERIES, N_KEYS,
                scale,
                ctx.D,
                ctx.Q_TILE_SIZE,
                ctx.K_TILE_SIZE,
                ctx.is_causal
            )

        
        ctx.save_for_backward(
            L, Q, K, V, O
        )

        return O

    @staticmethod
    def backward():
        raise NotImplementedError
