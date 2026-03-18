"""
TimeSliver Network for SeqComb-MV (Multivariate) classification.

Architecture:
- 2 CNN branches (4-size and 7-size motifs)
- Positional encoding
- AvgPool3d reduction
- Linear: 720 * max_m -> num_classes
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config


# Global variables for gradient hooks
cam = []
initial_out = []
initial_pool = []


def _hook_initial_q(grad):
    """Hook to capture gradients for initial output."""
    initial_out.append(grad)


def _hook_extract(grad):
    """Hook to capture gradients for CAM."""
    cam.append(grad)


class TimeSlicerDataset(Dataset):
    """Dataset class for TimeSliver model."""

    def __init__(self, ohe, sax, classes, seq_len, output, n_samples):
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.sax = torch.from_numpy(sax.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(np.int64))
        self.classes = torch.from_numpy(classes.astype(np.int64))
        self.output = torch.from_numpy(output.astype(np.int64)).reshape((n_samples,))
        self.n_samples = n_samples

    def __getitem__(self, index):
        return (
            self.ohe[index],
            self.sax[index],
            self.classes[index],
            self.seq_len[index],
            self.output[index],
        )

    def __len__(self):
        return self.n_samples


class PositionalEncoding(nn.Module):
    """Positional encoding module."""

    def __init__(self, d_model: int, dropout: float = 0.05, max_len: int = 18000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model % 2 != 0:
            div_term = torch.exp(
                torch.arange(0, d_model + 1, 2) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, 1, d_model + 1)
        else:
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, 1, d_model)

        position = torch.arange(max_len).unsqueeze(1)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        if d_model % 2 != 0:
            pe = pe[:, :, 0:-1]

        self.register_buffer("pe", pe)

    def forward(self, x, device):
        x = x + self.pe[: x.size(0)].to(device)
        return self.dropout(x)


class TimeSliverNetwork(nn.Module):
    """
    TimeSliver Network for SeqComb-MV classification.

    Uses 2 CNN branches with different kernel sizes to capture
    motifs at multiple scales.
    """

    def __init__(self, num_classes, d_model, d_out, max_m, device):
        super(TimeSliverNetwork, self).__init__()
        self.max_m = max_m
        self.device = device
        self.d_in = d_model
        self.d_model = d_model
        self.d_out = d_out
        sax_alpha = config.SAX_ALPHABET_SIZE

        # Projection layer
        self.proj = nn.Linear(self.d_in, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model)

        # CNN branch 1: kernel size 2 (4-size motifs)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(self.d_model, 16, 2, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, int(self.d_out), 2, stride=1),
        )
        self.ln1 = nn.LayerNorm(self.d_out)

        # CNN branch 2: kernel size 3 (7-size motifs)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(self.d_model, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, int(self.d_out), 3, stride=1),
        )
        self.ln2 = nn.LayerNorm(self.d_out)

        self.ln4 = nn.LayerNorm(sax_alpha)

        # Pooling layers
        self.maxpool1 = nn.Sequential(
            nn.AvgPool1d(2, stride=1),
            nn.AvgPool1d(2, stride=1),
            nn.AvgPool1d(2, stride=1),
        )

        self.maxpool2 = nn.Sequential(
            nn.AvgPool1d(3, stride=1),
            nn.AvgPool1d(3, stride=1),
            nn.AvgPool1d(3, stride=1),
        )

        # Motif sizes
        self.motif_size_1 = 4
        self.motif_size_2 = 7

        # Reduction layer
        self.reduction = nn.Sequential(nn.AvgPool3d((2, 1, 2), stride=(2, 1, 2)))

        # Classification layer
        self.nn = nn.Sequential(nn.Linear(int(720 * self.max_m), num_classes))

    def make_p(self, out, sax, seq_len, m):
        """Create motif-document interaction matrix."""
        cnn_net = getattr(self, f"cnn{m}")
        pool_net = getattr(self, f"maxpool{m}")
        ln_net = getattr(self, f"ln{m}")
        mo_size = getattr(self, f"motif_size_{m}")

        q = cnn_net(out)
        q = torch.permute(ln_net(torch.permute(q, (0, 2, 1))), (0, 2, 1))

        store_q = q

        d = pool_net(sax) * mo_size
        d = d / seq_len[0]

        q = torch.matmul(d, q.permute(0, 2, 1))
        q = q.reshape((q.size(0), q.size(1), q.size(2), 1))

        return store_q, d, q

    def forward(self, x, sax, seq_len):
        """Forward pass."""
        out = self.proj(x)

        out = self.positional_encoding(out.permute(1, 0, 2), self.device)
        out = out.permute(1, 2, 0)

        sax = sax.permute(0, 2, 1)

        _, _, out_1 = self.make_p(out, sax, seq_len, 1)
        _, _, out_2 = self.make_p(out, sax, seq_len, 2)

        heat_map = torch.cat((out_1, out_2), dim=3)
        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = self.reduction(heat_map)

        heat_map = nn.Flatten()(heat_map)
        heat_map = self.nn(heat_map)

        return heat_map

    def assigning_importance(self, mo_level, kernel_size, unwrapped_len):
        """Assign importance scores to sequence positions."""
        reduced_len = mo_level.size(-1)
        sequence_importance = torch.zeros((mo_level.size(0), unwrapped_len)).to(
            self.device
        )

        for i in range(reduced_len):
            sequence_importance[:, i : (i + kernel_size)] += mo_level[:, i].unsqueeze(
                -1
            )

        return sequence_importance

    @torch.no_grad()
    def calculate_motif_level(self, dp, m_i, initial_cam, P):
        """Calculate motif-level importance."""
        d_comp = getattr(self, f"pool_{m_i}")
        q_comp = getattr(self, f"out_{m_i}")

        total = d_comp.size(0)
        max_len = d_comp.size(-1)
        q_id = q_comp.size(1)
        d_id = d_comp.size(1)

        all_motif_importance = torch.zeros((total, max_len)).to(self.device)
        temp = torch.zeros((total, d_comp.size(-1))).to(self.device)
        var_imp = torch.zeros((total, 1)).to(self.device)
        signs = torch.zeros((total, 1)).to(self.device)

        for i in range(q_id):
            for j in range(d_id):
                var_imp[:, :] = (dp[:, j, i]).unsqueeze(-1)
                signs[:, :] = torch.sign(var_imp)

                temp[:, :] = nn.ReLU()(signs * d_comp[:, j, :] * q_comp[:, i, :])

                all_motif_importance += temp * abs(var_imp)

        return all_motif_importance

    def forward_motif_importance(self, x, sax, seq_len):
        """Forward pass with gradient hooks for importance calculation."""
        global cam, initial_out, initial_pool
        cam = []
        initial_out = []
        initial_pool = []

        self.Lin = x.size(1)
        self.seq_len = seq_len

        out = self.proj(x)

        out = self.positional_encoding(out.permute(1, 0, 2), self.device)
        out = out.permute(1, 2, 0)

        sax = sax.permute(0, 2, 1)

        self.out_1, self.pool_1, p_1 = self.make_p(out, sax, seq_len, 1)
        self.out_1.register_hook(_hook_initial_q)

        self.out_2, self.pool_2, p_2 = self.make_p(out, sax, seq_len, 2)

        heat_map = torch.cat((p_1, p_2), dim=3)
        heat_map.register_hook(_hook_extract)

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = self.reduction(heat_map)

        heat_map = nn.Flatten()(heat_map)
        heat_map = self.nn(heat_map)

        return heat_map, cam, initial_out, p_1
