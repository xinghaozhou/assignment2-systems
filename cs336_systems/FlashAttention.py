from jaxtyping import Float, Bool, Int
from torch import Tensor
from einops import rearrange
import torch
import math
import triton
from triton.language import tl

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_k"],
        is_causal=False
    ):

        # Q = [B, T_q, d_k]
        # K = [B, T_k, d_k]
        # V = [B, T_v, d_k]
    
        dtype = Q.dtype
        device = Q.device
        

        # initialize O in global memory
        # O is exactly like Q.shape
        O = torch.zeros_like(Q, device=device, dtype=dtype) # O = [B, T_q, d_k]

        # initialize L in global memory
        # Li is log (sum of exp(Sij) over rows)
        # Sij = (... queries, keys)
        # Li = (... queries)
        L = torch.zeros(Q.shape[:-1], device=device, dtype=dtype) # L = [B, T_q, ]

        batch = Q.size(0)
        d = Q.size(-1) # d_k

        Nq = Q.size(-2) # 
        Bq = 16 # Tile size

        Nk = K.size(-2) # B * T_k
        Bk = 16

        Tq = math.ceil(Nq / Bq) # num_row_tile
        Tk = math.ceil(Nk / Bk) # num_col_tile

        for i in range(1, Tq+1):
            # Dynamic (if Nq/Bq is not whole number)
            start_q = (i-1) * Bq
            end_q = min(i * Bq, Nq)

            Q_i = Q[..., start_q:end_q, :] # Q_i = [Bq_i, d_k]
            Bq_i = end_q - start_q # The size of tile_i in row

            O_i_orig = torch.zeros(size=(batch, Bq_i, d), dtype=dtype, device=device) # O_i_orig = [B, Bq_i, d_k]
            l_i_orig = torch.zeros(size=(batch, Bq_i, ), dtype=dtype, device=device) # l_i_orig = [B, Bq_i, ]
            m_i_orig = torch.full((batch, Bq_i, ), -float(torch.inf), dtype=dtype, device=device) # m_i_orig = [B, Bq_i, ]

            m_i_prev = m_i_orig # m_i_0
            l_i_prev = l_i_orig# l_i_0
            O_i_prev = O_i_orig # O_i_0

            for j in range(1, Tk+1):
                # Dynamic (if Nk/Bk is not whole number)
                start_k = (j-1) * Bk
                end_k = min(j * Bk, Nk)

                K_j = K[..., start_k:end_k, :] # K_j = [B, Bk_i, d_k]
                V_j = V[..., start_k:end_k, :] # V_j = [B, Bk_i, d_k]

                Bk_i = end_k - start_k

                # Calculate the pre-softmax attn scores
                Si_j = (Q_i @ K_j.permute(0, -1, -2)) / (d ** 0.5) # [B, Bq_i, d_k] @ [B, d_k, Bk_i] = [B, Bq_i, Bk_i]

                # Calculate the M here
                # mi_j = max attn score at each row (across each tile i)
                # use for value stability
                # mi_j = [Bq_i]
                mi_j = torch.max(m_i_prev, torch.max(Si_j, dim=-1)[0])

                # Pi_j = exp of stable attns of tile i j
                # Pi_j = [Bq_i, Bk_i]
                Pi_j = torch.exp(Si_j - mi_j[..., None])

                # Running update 
                # scaled previous denominator (scale * previous denom) + new tile denominator
                # li_j = [Bq_i, ] * [Bq_i] + [Bq_i, ]
                li_j = torch.exp(m_i_prev - mi_j) * l_i_prev + torch.sum(Pi_j, dim=-1)

                # Running update
                # scaled previous output (scale * previous output) + new tile output
                # Diag ensures that each query row is rescaled independently using its own max value, because m_i is different across rows.
                # Oi_j = [Bq_i, 1] * [Bq_i, d_k] + [Bq_i, Bk_i] @ [Bk_i, d_k]
                Oi_j = (torch.exp(m_i_prev - mi_j))[..., None] * O_i_prev + Pi_j @ V_j

                # Update for next iter
                m_i_prev = mi_j
                l_i_prev = li_j
                O_i_prev = Oi_j


            # When the iter ends, l_i_prev = l_i_Tk

            # (Bq, 1) @ (Bq, d_k) (1, d_k)
            O_i = 1/(l_i_prev[..., None]) * O_i_prev
            L_i = m_i_prev + torch.log(l_i_prev)

            O[..., start_q:end_q, :] = O_i
            L[..., start_q:end_q] = L_i

        ctx.save_for_backward(
            L, Q, K, V, O
        )
       
        return O

    @staticmethod
    def backward():
        raise NotImplementedError



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
        L_ptr + batch_index + stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(0, ),
        order=1
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

    Q_i = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option="zero") # (Q_TILE_SIZE, D)

    O_i_orig = tl.zeros(size=(Q_TILE_SIZE, D), dtype=tl.float32)
    l_i_orig = tl.zeros(size=(Q_TILE_SIZE, ), dtype=tl.float32)
    m_i_orig = tl.zeros(size=(Q_TILE_SIZE, ), dtype=tl.float32)

    O_i_prev = O_i_orig
    l_i_prev = l_i_orig
    m_i_prev = m_i_orig

    # Start the j loop
    for j in range(tl.cdiv(D, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, tl.trans(K_j)) / (D ** 0.5) # (Q_TILE_SIZE, K_TILE_SIZE)
        m_ij = tl.max(m_i_prev,  tl.max(S_ij, dim=-1)) # (Q_TILE_SIZE, )
        P_ij = tl.exp(S_ij - m_ij[..., None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        l_ij = tl.exp(m_i_prev - m_ij) * l_i_prev + tl.sum(P_ij, dim=-1) # (Q_TILE_SIZE, )
        O_ij = tl.exp(m_i_prev - m_ij)[..., None] * O_i_prev + tl.dot(P_ij, V_j) # (Q_TILE_SIZE, D)

        # Moving
        K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr.advance((0, K_TILE_SIZE))

        O_i_prev = O_ij
        l_i_prev = l_ij
        m_i_prev = m_ij

    O_i = tl.dot((1 / l_i_prev)[..., None], O_i_prev)
    L_i = m_i_prev + tl.log(l_i_prev)

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0, 1))

    Q_block_ptr.advance((Q_TILE_SIZE, 0))
    O_block_ptr.advance((Q_TILE_SIZE, 0))
    L_block_ptr.advance((Q_TILE_SIZE, 0))    

class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
        x,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_k"],
        is_causal=False
        ):

        breakpoint()
        raise NotImplementedError

    @staticmethod
    def backward():
        raise NotImplementedError
