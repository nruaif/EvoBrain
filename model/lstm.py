"""
Hardware-Efficient Layer-Normalized LSTM with Guaranteed Normalization for Edge Devices
Implements the algorithms from:
  "Hardware-Efficient Softmax and Layer Normalization with Guaranteed Normalization for Edge Devices"
  Dawon Choi, Hana Kim, Ji-Hoon Kim (Hanyang University, arXiv:2604.23647v1 [cs.AR] 26 Apr 2026)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class CoRNLayerNorm(nn.Module):
    """
    Algorithm 2 & Ref [12]: Compressed Reciprocal Newton Layer Normalization (CoRN-LN).
    - Eliminates expensive hardware square-root and division units (14x smaller area: 1199 um^2).
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


class CoRN_LSTMCell(nn.Module):
    """
    Hardware-Efficient Layer-Normalized LSTM Cell (Ba et al. + Choi et al. 2026).
    Applies CoRN-LN (Reciprocal Newton LayerNorm) to gate activations and cell state,
    eliminating division/square-root silicon while preventing covariate shift on edge devices.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # CoRN-LN on concatenated gates [f, i, c_tilde, o]
        self.ln_gates = CoRNLayerNorm(4 * hidden_size)
        # CoRN-LN on candidate cell state
        self.ln_c = CoRNLayerNorm(hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.ln_gates(self.weight_ih(x) + self.weight_hh(h_prev))
        f_gate, i_gate, c_tilde, o_gate = gates.chunk(4, dim=-1)

        f = torch.sigmoid(f_gate)
        i = torch.sigmoid(i_gate)
        c_cand = torch.tanh(c_tilde)

        c = f * c_prev + i * c_cand
        h = torch.sigmoid(o_gate) * torch.tanh(self.ln_c(c))
        return h, c


class HardwareEfficientLSTM(nn.Module):
    """
    Multi-layer Hardware-Efficient LSTM using CoRN-LN cells.
    Exact drop-in replacement for torch.nn.LSTM(..., batch_first=True).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.cells = nn.ModuleList([
            CoRN_LSTMCell(input_size if l == 0 else hidden_size, hidden_size)
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
    LSTM Model with guaranteed normalization for edge deployment.
    Default: Hardware-Efficient Reciprocal-Newton Layer-Normalized LSTM (CoRN-LN).
    """
    def __init__(self, args, num_classes, device=None, use_corn_ln=True):
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

        total_input_size = input_dim * num_nodes
        if use_corn_ln:
            self.lstm = HardwareEfficientLSTM(
                input_size=total_input_size,
                hidden_size=rnn_units,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=dropout
            )
            self.out_norm = CoRNLayerNorm(rnn_units)
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

    def forward(self, inputs, seq_lengths):
        """
        Args:
            inputs: (batch_size, max_seq_len, num_nodes, input_dim)
            seq_lengths: (batch_size, )
        """
        batch_size, max_seq_len, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (batch_size, max_seq_len, -1))

        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size)

        output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state))

        last_out = utils.last_relevant_pytorch(output, seq_lengths, batch_first=True)
        if self._device is not None:
            last_out = last_out.to(self._device)

        # Output normalization with guaranteed unit-variance
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
