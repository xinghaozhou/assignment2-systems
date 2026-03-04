from typing import Type, Float
from torch import Tensor
from einops import rearrange
import torch



class FlashAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_k"],
        is_causal=False
    ):
        dtype = Q.dtype
        device = Q.device

        Q = rearrange(Q, "... quries d_k -> (... queries) d_k")
        K = rearrange(K, "... keys d_k -> (... keys) d_k")
        V = rearrange(V, "... values d_k -> (... values) d_k")

        # initialize O and L in global memory

        # O is exactly like Q.shape
        O = torch.zeros_like(Q, device=device, dtype=dtype)

        # Li is log (sum of exp(Sij) over rows)
        # Sij = (... queries, keys)
        # Li = (... queries)
        L = torch.zeros_like(Q.squeeze(-1), device=device, dtype=dtype)

        d = Q.size(-1)

        Nq = Q.size(-2)
        Bq = 16 # Tile size

        Nk = K.size(-2)
        Bk = 16

        Tq = torch.ceiling(Nq / Bq)
        Tk = torch.ceiling(Nk / Bk)

        for i in range(1, Tq+1):
            Q_i = Q[(i-1) * Bq: i * Bq, :].copy()
            O_i_orig = torch.zeros(size=(Bq, d), dtype=dtype, device=device)
            l_i_orig = torch.zeros(size=(Bq, ), dtype=dtype, device=device)
            m_i_orig = -torch.inf(size=(Bq, ), dtype=dtype, device=device)

            m_i_prev = m_i_orig
            l_i_prev = l_i_orig
            O_i_prev = O_i_orig

            for j in range(1, Tk+1):

                K_j = K[(j-1) * Bk: j * Bk, :].copy()
                V_j = V[(j-1) * Bk: j * Bk, :].copy()

                # Calculate the pre-softmax attn scores
                Si_j = (Q_i @ K_j).T / (d ** 0.5)

                # Calculate the M here
                mi_j = torch.max(m_i_prev, torch.max(Si_j, dim=0))

                # calcualte P here
                Pi_j = torch.exp(Si_j - mi_j)

                # calculate L here
                li_j = torch.exp(m_i_prev - mi_j) @ l_i_prev + torch.sum(Pi_j, dim=0)

                # calculate O here
                Oi_j = torch.diag(torch.exp(m_i_prev - mi_j)) @ O_i_prev + Pi_j @ V_j

                # Update for next iter
                m_i_prev = mi_j
                l_i_prev = li_j
                O_i_prev = Oi_j


            # Question: If li_j here is not reference before assignment, then when the iter ends, li_j = li_Tk
            # Same as others
            O_i = torch.inverse(torch.diag(li_j)) @ Oi_j
            L_i = mi_j + torch.log(li_j)

            O[(i-1) * Bq: i * Bq, :] = O_i
            L[(i-1) * Bq: i * Bq, :] = L_i

        return O, L



    @staticmethod
    def backward():
        raise NotImplementedError


    
