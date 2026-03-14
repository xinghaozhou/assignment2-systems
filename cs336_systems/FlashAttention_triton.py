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

@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, D_ptr,
    dO_ptr, 
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq, 
    stride_dOb, stride_dOq, stride_dOd,
    stride_dQb, stride_dQq, stride_dQd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr=False
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        stride=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    ) # (N_QUERIES, D)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        stride=(stride_oq, stride_od),
        offset=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    ) # (N_QUERIES, D)

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dQb,
        shape=(N_QUERIES, D),
        stride=(stride_dQq, stride_dQd),
        offset=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dOb,
        shape=(N_QUERIES, D),
        stride=(stride_dOq, stride_dOd),
        offset=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    ) # (N_QUERIES, D)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        stride=(stride_lq, ),
        offset=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=0,
    ) # (N_QUERIES, )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, ),
        stride=(stride_dq, ),
        offset=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=0,
    ) # (N_QUERIES, )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    dQ_i = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    L_i = tl.load(L_block_ptr, boundary_check=(0), padding_options="zero") # (Q_TILE_SIZE, )
    D_i = tl.load(D_block_ptr, boundary_check=(0), padding_options="zero") # (Q_TILE_SIZE, )

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            stride=(stride_kk, stride_kd),
            offset=(j * K_TILE_SIZE, ),
            order=(1, 0)
        ) # (K_TILE_SIZE, D)

        V_j = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            stride=(stride_vk, stride_vd),
            offset=(j * K_TILE_SIZE, ),
            order=(1, 0)
        ) # (K_TILE_SIZE, D)

        dK_j = tl.zeros(size=(K_TILE_SIZE, D)) # (K_TILE_SIZE, D)

        dV_j = tl.zeros(size=(K_TILE_SIZE, D)) # (K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, K_j.permuate(-1, -2)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        P_ij = tl.exp((S_ij - L_i[..., None])) # (Q_TILE_SIZE, K_TILE_SIZE)

        dV_j += tl.dot(P_ij.permute(-1, -2), dO_i) # (K_TILE_SIZE, D)

        dP_ij = tl.dot(dO_i, V_j.permute(-1, -2)) # (Q_TILE_SIZE, K_TILE_SIZE)

        dS_ij = P_ij * (dP_ij - D_i[..., None]) / (D ** 0.5) # (Q_TILE_SIZE, K_TILE_SIZE)

        tl.atomic_add(dQ_i, tl.dot(dS_ij, K_j)) # Atomic add (Q_TILE_SIZE, K_TILE_SIZE)

    tl.store(dK_ptr, dK_j, boundary_check=(0, 1))
    tl.store(dV_ptr, dV_j, boundary_check=(0, 1))

    # Moving those in i-iteration
    Q_block_ptr.advance((Q_TILE_SIZE, 0))
    O_block_ptr.advance((Q_TILE_SIZE, 0))
    dO_block_ptr.advance((Q_TILE_SIZE, 0))
    L_block_ptr.advance((Q_TILE_SIZE,))
    D_block_ptr.advance((Q_TILE_SIZE,))

    return dQ_ptr, dK_ptr, dV_ptr
 

   


    

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
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        Q_TILE_SIZE, K_TILE_SIZE, is_causal = ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE, ctx.is_causal
        q_batch, N_QUERIES, Dk = Q.shape
        k_batch, N_KEYS, Dk = K.shape

        scale = 1 / (Dk ** 0.5)

        D = torch.sum(dO * O, dim=-1) # (B, N_QUERIES) 
        dQ = torch.zeros_like(Q) # (B, N_QUERIES, D)
        dK = torch.zeros_like(K) # (B, N_KEYS, D)
        dV = torch.zeros_like(V) # (B, N_KEYS, D)

        flash_bwd_kernel[triton.cdiv(N_QUERIES, Q_TILE_SIZE), q_batch](
            Q, K, V,
            O, L, D,
            dO,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),      # stride_qb, stride_qq, stride_qd
            K.stride(0), K.stride(1), K.stride(2),      # stride_kb, stride_kk, stride_kd
            V.stride(0), V.stride(1), V.stride(2),      # stride_vb, stride_vk, stride_vd
            O.stride(0), O.stride(1), O.stride(2),      # stride_ob, stride_oq, stride_od
            L.stride(0), L.stride(1),                   # stride_lb, stride_lq, stride_ld  
            D.stride(0), D.stride(1),                   # stride_db, stride_dq, 
            dO.stride(0), dO.stride(1), dO.stride(2),   # stride_dob, stride_doq, stride_dod
            dQ.stride(0), dQ.stride(1), dQ.stride(2),   # stride_dqb, stride_dqq, stride_dqd
            dK.stride(0), dK.stride(1), dK.stride(2),   # stride_dkb, stride_dkk, stride_dkd
            dV.stride(0), dV.stride(1), dV.stride(2),   # stride_dvb, stride_dvk, stride_dvd
            N_QUERIES, N_KEYS,
            scale,
            ctx.D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            ctx.is_causal
        )

        return 









