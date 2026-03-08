from jaxtyping import Float, Bool, Int
from torch import Tensor
from einops import rearrange
import torch
import math

class FlashAttentionPytorch(torch.autograd.Function):
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
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        B, N_QUERIES, d = Q.shape 

        B, N_KEYS, d = K.shape

        dtype = Q.dtype

        Bq = 16
        Bk = 16

        Tq = math.ceil(N_QUERIES / Bq)
        Tk = math.ceil(N_KEYS / Bk)

        # only need to store D, L 
        D = torch.sum(dO * O, dim=-1) # (B, N_QUERIES) 
        L = torch.zeros((B, N_QUERIES, ), dtype=dtype) # (B, N_QUERIES)
        dQ = torch.zeros_like(Q) # (B, N_QUERIES, D)
        dK = torch.zeros_like(K) # (B, N_KEYS, D)
        dV = torch.zeros_like(V) # (B, N_KEYS, D)
        
        for j in range(1, Tk):
            start_k = (i-1) * Bk
            end_k = min(i * Bk, N_KEYS)
            
            K_j = K[..., start_k:end_k, :] # (B, K_TILE_SIZE, D)
            V_j = V[..., start_k:end_k, :] # (B, K_TILE_SIZE, D)

            dK_j = torch.zeros((B, Bk, d), dtype=dtype)
            dV_j = torch.zeros((B, Bk, d), dtype=dtype)

            for i in range(1, Tq):
                start_q = (i-1) * Bq
                end_q = min(i * Bq, N_QUERIES)

                Q_i = Q[..., start_q:end_q, :] # (B, Q_TILE_SIZE, D)
                O_i = O[..., start_q:end_q, :] # (B, Q_TILE_SIZE, D)
                dO_i = dO[..., start_q:end_q, :] # (B, Q_TILE_SIZE, D)
                dQ_i = dQ[..., start_q:end_q, :] # (B, Q_TILE_SIZE)

                L_i = L[..., start_q:end_q] # (B, Q_TILE_SIZE, )
                D_i = D[..., start_q:end_q] # (B, Q_TILE_SIZE, )

                S_ij = (Q_i @ K_j.permute(0, -1, -2)) / (d ** 0.5) # (B, Q_TILE_SIZE, K_TILE_SIZE)

                P_ij = torch.exp(S_ij - L_i[..., None]) # (B, Q_TILE_SIZE, K_TILE_SIZE)

                dV_j += (P_ij.permute(0, -1, -2)) @ dO_i # (B, K_TILE_SIZE, )

                dP_ij = dO_i @ V_j.permute(0, -1, -2) # (B, Q_TILE_SIZE, K_TILE_SIZE) 
                dS_ij = P_ij * (dP_ij - D_i[..., None]) / (d ** 0.5) # (B, Q_TILE_SIZE, K_TILE_SIZE) 

                dQ[..., start_q:end_q, :] = dQ_i
                dQ_i += dS_ij @ K_j # (B, Q_TILE_SIZE, D)
                
                dK_j += dS_ij.permute(0, -1, -2) @ Q_i # (B, K_TILE_SIZE, D)

            dK[..., start_k:end_k, :] = dK_j
            dV[..., start_k:end_k, :] = dV_j

        return dQ, dK, dV, None
