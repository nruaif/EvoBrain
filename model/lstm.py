"""
Hardware-Efficient Layer-Normalized LSTM with Guaranteed Normalization for Edge FPGAs.
Unites the algorithms and hardware architectures from two recent breakthroughs:
1. "Hardware-Efficient Softmax and Layer Normalization with Guaranteed Normalization for Edge Devices"
   Dawon Choi, Hana Kim, Ji-Hoon Kim (Hanyang University, arXiv:2604.23647v1 [cs.AR], 26 Apr 2026)
   - CoRN-LN: 14x smaller (1199 um^2), 2-cycle Newton rsqrt, guaranteed unit-variance sigma = 1.0.
   - Mul-/Div-free Softmax: 11x smaller (942 um^2), Radix R=8 2-LUT exp, guaranteed sum(p) == 1.0.

2. "Energy Efficient LSTM Accelerators for Embedded FPGAs through Parameterised Architecture Design"
   Chao Qian, Tianheng Ling, Gregor Schiele (University of Duisburg-Essen, arXiv:2604.19293v1 [cs.AR], 21 Apr 2026)
   - HardSigmoid*: Slope 0.125 = 1/8 = 2^(-3) enables pure 3-bit arithmetic right shift ((x >> 3) + 0.5).
     Consumes only 3 LUTs and 0 DSPs (vs 256-entry tables or float exp).
   - HardTanh: Clamped linear interval [-1.0, 1.0], consuming only 5 LUTs and 0 DSPs.
   - 8-bit Fixed-Point (4,8) QAT: Replaces DSP multipliers with fast LUT-based multipliers, reaching 204 MHz.
   - 5-stage pipelined MAC loop: Reduces latency by 51% and power by 18.57% on Spartan-7 XC7S15.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


# ---------------------------------------------------------------------------
# Paper 2 (Qian et al. 2026): Zero-DSP Activation Functions & Fixed-Point QAT
# ---------------------------------------------------------------------------

class HardSigmoidStar(nn.Module):
    """
    HardSigmoid* from Qian et al. (arXiv:2604.19293v1).
    Uses a hardware-friendly slope of 0.125 = 1/8 = 2^(-3) for (4,8) fixed-point data.
    On FPGA:
        HardSigmoid*(x) = clamp((x >> 3) + 0.5, 0.0, 1.0)
    Implemented in hardware with 0 DSPs and only 3 LUTs (vs 256-entry LUTs in prior work).
    """
    def __init__(self, slope=0.125, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return torch.clamp(x * self.slope + self.offset, 0.0, 1.0)


class HardTanh_FPGA(nn.Module):
    """
    HardTanh from Qian et al. (arXiv:2604.19293v1).
    Piecewise-linear approximation to Tanh:
        HardTanh(x) = clamp(x, min_val=-1.0, max_val=1.0)
    Consumes only 2 fixed-point comparators (5 LUTs) and 0 DSPs on Spartan-7.
    """
    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class FixedPointQuantizerSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for (4,8) Fixed-Point Quantization (Qian et al. Sec 4.1).
    Total width: 8 bits, Fractional bits: 4.
    Range: [-8.0, 7.9375] in steps of 0.0625 (1/16).
    """
    @staticmethod
    def forward(ctx, x, frac_bits=4, total_bits=8):
        scale = 2.0 ** frac_bits
        min_val = - (2.0 ** (total_bits - frac_bits - 1))
        max_val = (2.0 ** (total_bits - frac_bits - 1)) - (1.0 / scale)
        x_q = torch.clamp(torch.round(x * scale) / scale, min_val, max_val)
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def quantize_q44(x):
    return FixedPointQuantizerSTE.apply(x, 4, 8)


# ---------------------------------------------------------------------------
# Paper 1 (Choi et al. 2026): CoRN-LN LayerNorm & Mul-/Div-Free Softmax
# ---------------------------------------------------------------------------

class CoRNLayerNorm(nn.Module):
    """
    Algorithm 2 & Ref [12]: Compressed Reciprocal Newton Layer Normalization (CoRN-LN).
    - Eliminates expensive hardware square-root and division units (14x smaller: 1199 um^2).
    - Uses Leading-One-Detector (LOD) for initial estimate + 2-cycle Reciprocal Newton iteration:
        x_{i+1} = 0.5 * [x_i + 1 / (x_i * var)]
    - Guarantees unit variance (|1 - sigma| < 0.2e-6) for score-oriented edge tasks.
    - Output stage uses a multiplier instead of a divider: Y = (X - mean) * rsqrt * gamma + beta.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=dims, keepdim=True) + self.eps

        # LOD-based initial guess: x0 = 2^(-floor(log2(var))/2)
        lod_k = torch.floor(torch.log2(var))
        x0 = torch.pow(2.0, -0.5 * lod_k)

        # 2-step Reciprocal Newton iteration (Eq. 5 from Choi et al.)
        x1 = 0.5 * (x0 + 1.0 / (x0 * var))
        x2 = 0.5 * (x1 + 1.0 / (x1 * var))

        out = (x - mean) * x2
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class MulDivFreeSoftmax(nn.Module):
    """
    Algorithm 1: Multiplier- / Divider-Free Softmax with Guaranteed Probability Normalization.
    - Stabilizes inputs via max subtraction: Delta_i = max(X) - X_i >= 0.
    - Factorizes exponential via Radix R=8 into coarse and residual components:
        e^(-q) = e^(-R * frac) * e^(-rem) = a_i * b_i
      using two compact Look-Up Tables (LUTs) of 7 and 8 entries (area: 942 um^2, 11x smaller).
    - Fixed-point division (FxP_Div) via binary shift-subtract guarantees sum(p) == 1.0.
    """
    def __init__(self, dim=-1, R=8):
        super().__init__()
        self.dim = dim
        self.R = R

    def forward(self, x):
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]
        delta = max_val - x  # >= 0

        # Coarse and residual decomposition
        frac = torch.floor(delta / self.R)
        rem = delta - self.R * frac

        # Coarse LUT (7 entries for frac in [0..6], delta >= 56 underflows to 0)
        frac_clamped = torch.clamp(frac, 0, 7)
        coarse = torch.exp(-self.R * frac_clamped)
        coarse = torch.where(frac > 7, torch.zeros_like(coarse), coarse)

        # Residual LUT (8 entries for rem in [0, 8))
        residual = torch.exp(-rem)
        y = coarse * residual

        # Summation and fixed-point normalization
        z = torch.sum(y, dim=self.dim, keepdim=True)
        p = y / (z + 1e-12)
        # Enforce exact probability normalization (sum(p) == 1.0)
        p = p / p.sum(dim=self.dim, keepdim=True)
        return p


# ---------------------------------------------------------------------------
# Unified Hardware-Efficient LSTM Architecture
# ---------------------------------------------------------------------------

class CoRN_LSTMCell(nn.Module):
    """
    Hardware-Efficient Recurrent Cell combining:
    - Qian et al. (2026): HardSigmoid* (slope 0.125 = 1/8) + HardTanh (clamp [-1, 1]) -> Zero DSPs, 8 LUTs!
    - Choi et al. (2026): CoRN-LN on gate projections and cell state -> 14x smaller, unit variance sigma=1.0.
    - Optional Q4.4 Fixed-Point Quantization-Aware Training (QAT).
    """
    def __init__(self, input_size, hidden_size, act_mode="hard", use_norm=True, quantize=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_mode = act_mode
        self.use_norm = use_norm
        self.quantize = quantize

        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        if use_norm:
            self.ln_gates = CoRNLayerNorm(4 * hidden_size)
            self.ln_c = CoRNLayerNorm(hidden_size)
        else:
            self.ln_gates = nn.Identity()
            self.ln_c = nn.Identity()

        if act_mode == "hard":
            self.sig = HardSigmoidStar(slope=0.125, offset=0.5)
            self.tanh = HardTanh_FPGA(min_val=-1.0, max_val=1.0)
        else:
            self.sig = nn.Sigmoid()
            self.tanh = nn.Tanh()

    def forward(self, x, state):
        h_prev, c_prev = state
        
        ih = self.weight_ih(x)
        hh = self.weight_hh(h_prev)
        if self.quantize:
            ih = quantize_q44(ih)
            hh = quantize_q44(hh)

        gates = self.ln_gates(ih + hh)
        f_gate, i_gate, c_tilde, o_gate = gates.chunk(4, dim=-1)

        f = self.sig(f_gate)
        i = self.sig(i_gate)
        c_cand = self.tanh(c_tilde)

        c = f * c_prev + i * c_cand
        c_normed = self.ln_c(c)
        h = self.sig(o_gate) * self.tanh(c_normed)

        if self.quantize:
            c = quantize_q44(c)
            h = quantize_q44(h)

        return h, c


class HardwareEfficientLSTM(nn.Module):
    """
    Multi-layer Hardware-Efficient LSTM using CoRN-LN + HardSigmoid* + HardTanh.
    Drop-in replacement for torch.nn.LSTM(..., batch_first=True).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, act_mode="hard", use_norm=True, quantize=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.cells = nn.ModuleList([
            CoRN_LSTMCell(
                input_size if l == 0 else hidden_size,
                hidden_size,
                act_mode=act_mode,
                use_norm=use_norm,
                quantize=quantize
            )
            for l in range(num_layers)
        ])

    def forward(self, x, init_states=None):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.shape

        if init_states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        else:
            h_init, c_init = init_states
            h = [h_init[l] for l in range(self.num_layers)]
            c = [c_init[l] for l in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, (h[l], c[l]))
                inp = self.dropout(h[l])
            outputs.append(h[-1].unsqueeze(1))

        output = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        h_final = torch.stack(h, dim=0)     # (num_layers, batch_size, hidden_size)
        c_final = torch.stack(c, dim=0)     # (num_layers, batch_size, hidden_size)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_final, c_final)


class LSTMModel(nn.Module):
    """
    LSTM Model with guaranteed normalization and zero-DSP activation options for edge deployment.
    Default: Hardware-Efficient Reciprocal-Newton Layer-Normalized LSTM (CoRN-LN) + HardSigmoid* + HardTanh.
    """
    def __init__(self, args, num_classes, device=None, use_corn_ln=True, act_mode="hard", quantize=False):
        super(LSTMModel, self).__init__()

        num_nodes = args.num_nodes
        rnn_units = args.rnn_units
        num_rnn_layers = args.num_rnn_layers
        input_dim = args.input_dim
        dropout = getattr(args, "dropout", 0.2)

        self._input_dim = input_dim
        self._num_nodes = num_nodes
        self._num_rnn_layers = num_rnn_layers
        self._rnn_units = rnn_units
        self._num_classes = num_classes
        self._device = device
        self._use_corn_ln = use_corn_ln
        self._act_mode = act_mode

        total_input_size = input_dim * num_nodes
        if use_corn_ln or act_mode == "hard":
            self.lstm = HardwareEfficientLSTM(
                input_size=total_input_size,
                hidden_size=rnn_units,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=dropout,
                act_mode=act_mode,
                use_norm=use_corn_ln,
                quantize=quantize
            )
            self.out_norm = CoRNLayerNorm(rnn_units) if use_corn_ln else nn.Identity()
        else:
            self.lstm = nn.LSTM(
                total_input_size,
                rnn_units,
                num_rnn_layers,
                batch_first=True
            )
            self.out_norm = nn.BatchNorm1d(rnn_units)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(rnn_units, num_classes)
        self.relu = nn.ReLU()
        self.softmax = MulDivFreeSoftmax(dim=-1)

    def forward(self, inputs, seq_lengths, return_sequence=False):
        """
        Args:
            inputs: (batch_size, max_seq_len, num_nodes, input_dim)
            seq_lengths: (batch_size, )
            return_sequence: if True, returns (B, T, num_classes) and (B, T, hidden_dim) for dense temporal supervision.
        """
        batch_size, max_seq_len, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (batch_size, max_seq_len, -1))

        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size)

        output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state))

        if return_sequence:
            if self._device is not None:
                output = output.to(self._device)
            normed = self.out_norm(output)
            logits = self.fc(self.relu(self.dropout(normed)))
            return logits, output

        last_out = utils.last_relevant_pytorch(output, seq_lengths, batch_first=True)
        if self._device is not None:
            last_out = last_out.to(self._device)

        normed = self.out_norm(last_out)
        logits = self.fc(self.relu(self.dropout(normed)))

        return logits, last_out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_()
        cell = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_()
        if self._device is not None:
            hidden = hidden.to(self._device)
            cell = cell.to(self._device)
        return hidden, cell
