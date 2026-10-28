import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Zero-DSP Activations & APoT Quantizer
# ---------------------------------------------------------------------------
def quantize_to_apot(w, num_bits=4):
    """
    Additive Powers-of-Two (APoT):
    Each weight is represented as w = sign * (2^p1 + 2^p2) or {0}.
    Zero DSP multipliers on FPGA: multiplication is strictly two bit-shifts and an adder.
    """
    abs_w = w.abs()
    max_val = abs_w.max().clamp(min=1e-8)
    max_exp = torch.ceil(torch.log2(max_val))
    n_levels = (1 << (num_bits - 1)) - 1
    min_exp = max_exp - n_levels + 1
    zero_thresh = 2.0 ** (min_exp - 1)

    # First term: 2^p1
    sign = torch.sign(w)
    is_zero = abs_w < zero_thresh
    exp1 = torch.clamp(torch.round(torch.log2(abs_w.clamp(min=1e-12))), min=min_exp, max=max_exp)
    w1 = torch.where(is_zero, torch.zeros_like(w), sign * (2.0 ** exp1))

    # Second term: 2^p2 (residual)
    residual = w - w1
    abs_r = residual.abs()
    is_r_zero = abs_r < (zero_thresh / 2.0)
    exp2 = torch.clamp(torch.round(torch.log2(abs_r.clamp(min=1e-12))), min=min_exp - 2, max=max_exp - 1)
    w2 = torch.where(is_r_zero, torch.zeros_like(residual), torch.sign(residual) * (2.0 ** exp2))

    w_apot = w1 + w2
    # Straight-Through Estimator (STE)
    return (w_apot - w).detach() + w


class APoTLinear(nn.Module):
    """Linear layer using discrete Additive Powers-of-Two (APoT) weights (0 DSPs)."""
    def __init__(self, in_features, out_features, bias=True, num_bits=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w_apot = quantize_to_apot(self.weight, num_bits=self.num_bits)
        return F.linear(x, w_apot, self.bias)


# ---------------------------------------------------------------------------
# 2. Recurrent Cell (Per-Node Temporal Processing with APoT Weights)
# ---------------------------------------------------------------------------
class APoT_GRUCell(nn.Module):
    """Hardware-efficient GRU cell using APoT weights and Zero-DSP Hard activations."""
    def __init__(self, input_size=100, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = APoTLinear(input_size, 3 * hidden_size, bias=True)
        self.h2h = APoTLinear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x, h):
        # x: (B * N, input_size), h: (B * N, hidden_size)
        gates = self.x2h(x) + self.h2h(h)
        z, r, n = gates.chunk(3, dim=-1)

        # Zero-DSP activations: HardSigmoid* (x >> 3 + 0.5) and HardTanh (clamp [-1, 1])
        z = torch.clamp(0.125 * z + 0.5, 0.0, 1.0)
        r = torch.clamp(0.125 * r + 0.5, 0.0, 1.0)
        n = torch.clamp(n, -1.0, 1.0)

        h_new = (1.0 - z) * n + z * h
        return h_new


# ---------------------------------------------------------------------------
# 3. Spatial Graph Convolution (APoT GCN Layer)
# ---------------------------------------------------------------------------
class APoT_GCNLayer(nn.Module):
    """Spatial Graph Convolution: H' = ReLU(A_norm @ H @ W_apot)."""
    def __init__(self, in_dim=64, out_dim=64):
        super().__init__()
        self.proj = APoTLinear(in_dim, out_dim, bias=True)

    def forward(self, h, adj):
        # h: (B, N, in_dim), adj: (B, N, N)
        # Degree normalization: D^-1 A
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        adj_norm = adj / deg

        # Message passing: A_norm @ h
        msg = torch.bmm(adj_norm, h)  # (B, N, in_dim)
        out = self.proj(msg)          # (B, N, out_dim)
        return F.relu(out)


# ---------------------------------------------------------------------------
# 4. Full Standalone Model: EvoBrain APoT Multi-Task Architecture
# ---------------------------------------------------------------------------
class EvoBrainAPoTModel(nn.Module):
    """
    Self-Contained Full Forward Pass Model:
    - Input: EEG clip (Batch, 12 seconds, 19 electrodes, 100 FFT features)
    - Graph: Dynamic 19x19 Adjacency Matrix
    - Weights: 100% Additive Powers-of-Two (Zero DSP Multipliers)
    - Output: Dual heads (5 Time Horizons + 5 Seizure Subtypes)
    """
    def __init__(self, num_nodes=19, input_dim=100, hidden_dim=64, num_classes=5):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1. Temporal Recurrent Engine (Node-wise over 12 seconds)
        self.rnn_cell = APoT_GRUCell(input_size=input_dim, hidden_size=hidden_dim)

        # 2. Spatial Graph Engine (2-Layer GCN across 19 electrodes)
        self.gcn1 = APoT_GCNLayer(in_dim=hidden_dim, out_dim=hidden_dim)
        self.gcn2 = APoT_GCNLayer(in_dim=hidden_dim, out_dim=hidden_dim)

        # 3. Dual Linear Heads (Folded direct projections)
        self.time_head = APoTLinear(hidden_dim, 5, bias=True)           # [now, 5s, 10s, 20s, 40s]
        self.class_head = APoTLinear(hidden_dim, num_classes, bias=True) # [bckg, fnsz, gnsz, cpsz, other]

    def forward(self, x, adj):
        """
        Args:
            x: (B, 12, 19, 100) - 12-second EEG clips with 19 electrodes and 100 features/sec
            adj: (B, 19, 19) or (B, 12, 19, 19) - Dynamic correlation graph
        Returns:
            time_logits: (B, 5) - Proximity warning logits for [now, 5s, 10s, 20s, 40s]
            class_logits: (B, 5) - Seizure subtype logits for [bckg, fnsz, gnsz, cpsz, other]
        """
        B, T, N, D = x.shape  # B=Batch, T=12 seconds, N=19 channels, D=100 FFT features
        assert N == self.num_nodes and D == self.input_dim

        # If adj has time dimension (B, 12, 19, 19), average or take final graph
        if adj.dim() == 4:
            adj = adj.mean(dim=1)  # (B, 19, 19)

        # Step 1: Temporal GRU Processing over T=12 time steps
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)
        # Reshape x to iterate over time: (T, B * N, D)
        x_seq = x.permute(1, 0, 2, 3).reshape(T, B * N, D)
        for t in range(T):
            h = self.rnn_cell(x_seq[t], h)

        # Reshape to (B, N, hidden_dim)
        node_feats = h.view(B, N, self.hidden_dim)

        # Step 2: Spatial Topological GCN Message Passing
        gcn_out = self.gcn1(node_feats, adj)
        gcn_out = self.gcn2(gcn_out, adj)  # (B, 19, 64)

        # Step 3: Global Electrode Pooling (Max-Pooling across 19 nodes)
        clip_feat = torch.max(gcn_out, dim=1)[0]  # (B, 64)

        # Step 4: Dual Multi-Task Prediction Heads
        time_logits = self.time_head(clip_feat)    # (B, 5)
        class_logits = self.class_head(clip_feat)  # (B, 5)

        return time_logits, class_logits

# Smoke test
if __name__ == "__main__":
    model = EvoBrainAPoTModel()
    dummy_x = torch.randn(4, 12, 19, 100)
    dummy_adj = torch.rand(4, 19, 19)
    t_logits, c_logits = model(dummy_x, dummy_adj)
    print("Smoke test passed!")
    print("time_logits shape:", t_logits.shape)
    print("class_logits shape:", c_logits.shape)
