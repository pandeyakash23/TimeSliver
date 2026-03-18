"""
TimeSliver Network Architecture for Audio Classification.

This module contains:
- TimeSlicerDataset: PyTorch Dataset for audio data
- PositionalEncoding: Sinusoidal positional encoding (not used in forward)
- TimeSliverNetwork: Main classification network with single CNN branch

Audio-specific architecture:
- Single CNN branch with kernel size 1
- NO positional encoding in forward pass
- AvgPool2d((1, 3)) reduction
- Linear layer: d_out * 133 * max_m -> num_classes
"""
import torch
import numpy as np
import math
from torch.utils.data import Dataset
import torch.nn as nn

import config


# Global lists for gradient hooks
cam = []
initial_out = []
initial_pool = []


class TimeSlicerDataset(Dataset):
    """PyTorch Dataset for TimeSliver audio data."""

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
    """Sinusoidal positional encoding (not used in audio forward pass)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 3200):
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

    def forward(self, x, rank):
        x = x + self.pe[: x.size(0)].to(rank)
        return self.dropout(x)


def initial_q(grad):
    """Hook function to capture q gradients."""
    initial_out.append(grad)


def initial_d(grad):
    """Hook function to capture d gradients."""
    initial_pool.append(grad)


def extract(grad):
    """Hook function to extract gradients."""
    cam.append(grad)


class TimeSliverNetwork(nn.Module):
    """
    TimeSliver network for audio classification.

    Architecture:
    - Linear projection: d_in -> d_model
    - Single CNN branch with kernel=1 (no motif extraction, point-wise)
    - SAX pooling with kernel=1
    - Motif-document interaction matrix
    - AvgPool2d reduction
    - Linear classifier

    Audio-specific:
    - NO positional encoding
    - Single CNN branch (cnn2)
    - Reduction: AvgPool2d((1, 3))
    """

    def __init__(self, num_classes, d_model, d_out, max_m, rank):
        super(TimeSliverNetwork, self).__init__()
        self.max_m = max_m
        self.rank = rank
        self.d_in = config.D_IN
        self.d_model = d_model
        self.d_out = d_out
        sax_alpha = config.SAX_ALPHABET_SIZE
        self.max_len = config.MAX_LEN

        # Projection layer
        self.proj = nn.Linear(self.d_in, self.d_model)

        # Positional encoding (defined but not used in forward)
        self.positional_encoding = PositionalEncoding(self.d_model)

        # Single CNN branch with kernel=1 (point-wise operations)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(self.d_model, 16, 1, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, int(self.d_out), 1, stride=1),
        )
        self.ln2 = nn.LayerNorm(self.d_out)

        # Layer normalization for SAX
        self.ln4 = nn.LayerNorm([sax_alpha])
        self.start = nn.LayerNorm(self.d_in)

        # SAX pooling with kernel=1
        self.maxpool2 = nn.Sequential(
            nn.AvgPool1d(1, stride=1),
            nn.AvgPool1d(1, stride=1),
            nn.AvgPool1d(1, stride=1),
        )

        # Motif size for single branch
        self.motif_size_2 = 1 * 1

        # Reduction pooling
        self.reduction = nn.Sequential(
            nn.AvgPool2d((1, 3), stride=(1, 3)),
        )

        # Final classifier
        # Linear input = d_out * 133 * max_m
        self.nn = nn.Sequential(
            nn.Linear(int(self.d_out * config.LINEAR_INPUT_MULTIPLIER * self.max_m), num_classes),
        )

    def make_p(self, out, sax, seq_len, m):
        """Create motif-document interaction for branch m."""
        cnn_net = getattr(self, f"cnn{m}")
        pool_net = getattr(self, f"maxpool{m}")
        ln_net = getattr(self, f"ln{m}")
        mo_size = getattr(self, f"motif_size_{m}")

        q = cnn_net(out)  # [N, f, L]
        q = torch.permute(ln_net(torch.permute(q, (0, 2, 1))), (0, 2, 1))

        store_q = q

        # SAX pooling
        d = pool_net(sax) * mo_size  # [N, f, L]
        d = d / seq_len[0]

        # Motif-document interaction
        q = torch.matmul(d, q.permute(0, 2, 1))
        q = q.reshape((q.size(0), q.size(1), q.size(2), 1))

        return store_q, d, q

    def forward(self, x, sax, seq_len):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_in]
            sax: SAX encoded tensor [batch, seq_len, sax_alpha]
            seq_len: Sequence lengths [batch]

        Returns:
            Logits [batch, num_classes]
        """
        # Projection
        out = self.proj(x)

        # NOTE: No positional encoding for audio dataset

        out = torch.permute(out, (0, 2, 1))  # [N, d_model, L]
        sax = sax.permute(0, 2, 1)

        # Single CNN branch
        _, _, out_2 = self.make_p(out, sax, seq_len, 2)

        heat_map = out_2
        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = self.ln4(heat_map)

        heat_map = heat_map.squeeze()
        heat_map = self.reduction(heat_map)

        heat_map = nn.Flatten()(heat_map)
        heat_map = self.nn(heat_map)

        return heat_map

    def assigning_importance(self, mo_level, kernel_size, unwrapped_len):
        """Map motif importance back to sequence positions."""
        reduced_len = mo_level.size(-1)
        sequence_importance = torch.zeros((mo_level.size(0), unwrapped_len)).to(self.rank)

        for i in range(reduced_len):
            sequence_importance[:, i : (i + kernel_size)] += mo_level[:, i].unsqueeze(-1)

        return sequence_importance

    def sum_subsequent_n(self, temp, n):
        """Sum n subsequent elements using sliding window."""
        if n > len(temp):
            raise ValueError("n should be less than or equal to the length of temp")

        windowed_view = np.lib.stride_tricks.sliding_window_view(temp, n)
        return windowed_view.sum(axis=1)

    @torch.no_grad()
    def calculate_motif_level(self, dp, m_i, initial_cam):
        """Calculate motif-level importance from gradients."""
        d_comp = getattr(self, f"pool_{m_i}")
        q_comp = getattr(self, f"out_{m_i}")
        m_size = initial_cam.size(1) - (d_comp.size(-1) - 1)

        total = d_comp.size(0)
        max_len = d_comp.size(-1)
        q_id = q_comp.size(1)
        d_id = d_comp.size(1)

        all_motif_importance = torch.zeros((total, max_len)).to(self.rank)
        temp = torch.zeros((total, d_comp.size(-1))).to(self.rank)
        max_val_prot = torch.zeros((total, 1)).to(self.rank)
        var_imp = torch.zeros((total, 1)).to(self.rank)
        signs = torch.zeros((total, 1)).to(self.rank)

        for i in range(q_id):
            for j in range(d_id):
                var_imp[:, :] = dp[:, j, i].unsqueeze(-1)
                signs[:, :] = torch.sign(var_imp)

                temp[:, :] = nn.ReLU()(
                    signs * d_comp[:, j, :] * q_comp[:, i, :] * initial_cam[:, i, :]
                )

                max_val_prot[:, 0] = torch.max(temp, dim=1)[0].squeeze()
                temp[:, :] = (temp) / (max_val_prot + 1e-18)
                all_motif_importance += temp * abs(var_imp)

        return all_motif_importance

    def forward_motif_importance(self, x, sax, seq_len):
        """
        Forward pass with gradient hooks for attribution calculation.

        Returns:
            logits, cam gradients, initial_cam, heat_map
        """
        global cam, initial_out, initial_pool
        cam = []
        initial_out = []
        initial_pool = []
        self.Lin = x.size(1)
        self.seq_len = seq_len

        # Projection
        out = self.proj(x)

        # NOTE: No positional encoding for audio dataset

        out = torch.permute(out, (0, 2, 1))  # [N, d_model, L]
        sax = sax.permute(0, 2, 1)

        # Single CNN branch with hooks
        self.out_1, self.pool_1, p_1 = self.make_p(out, sax, seq_len, 2)
        self.out_1.register_hook(initial_q)

        heat_map = p_1
        heat_map.register_hook(extract)

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.squeeze()
        heat_map = self.reduction(heat_map)

        heat_map = nn.Flatten()(heat_map)
        heat_map = self.nn(heat_map)

        return heat_map, cam, initial_out, p_1
