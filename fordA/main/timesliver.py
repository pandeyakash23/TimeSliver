"""
TimeSliver Network Architecture.
CNN-based model with temporal attribution capabilities for time series classification.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class TimeSlicerDataset(Dataset):
    """Dataset class for TimeSliver model."""

    def __init__(self, ohe, sax, classes, seq_len, output, n_samples):
        """
        Initialize the dataset.

        Args:
            ohe: One-hot encoded input [samples, seq_len, features]
            sax: SAX encoded input [samples, seq_len, sax_alphabet]
            classes: Argmax of OHE [samples, seq_len]
            seq_len: Sequence lengths [samples]
            output: Labels [samples]
            n_samples: Number of samples
        """
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
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, d_model, dropout=0.1, max_len=3200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Handle odd/even dimensionality
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
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [seq_len, batch_size, embedding_dim]
            device: Device to move positional encoding to
        """
        x = x + self.pe[: x.size(0)].to(device)
        return self.dropout(x)


class TimeSliverNetwork(nn.Module):
    """
    TimeSliver Network for time series classification.

    Combines CNN-based motif extraction with SAX pooling for
    interpretable time series classification.
    """

    def __init__(self, num_classes, d_model, d_out, max_m, device):
        """
        Initialize the network.

        Args:
            num_classes: Number of output classes
            d_model: Model dimension (input feature dimension)
            d_out: Output dimension for CNN (default: 36)
            max_m: Number of motif scales
            device: Device to run on
        """
        super().__init__()
        self.max_m = max_m
        self.device = device
        self.d_in = d_model
        self.d_model = d_model
        self.d_out = d_out
        self.sax_alpha = 70

        # Projection layer
        self.proj = nn.Linear(self.d_in, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model)

        # CNN layers for motif extraction
        self.cnn1 = nn.Sequential(
            nn.Conv1d(self.d_model, 16, 4, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 4, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, int(self.d_out), 4, stride=1),
        )
        self.ln1 = nn.LayerNorm(self.d_out)

        # SAX pooling layers
        self.ln4 = nn.LayerNorm(self.sax_alpha)
        self.maxpool1 = nn.Sequential(
            nn.AvgPool1d(4, stride=1),
            nn.AvgPool1d(4, stride=1),
            nn.AvgPool1d(4, stride=1),
        )
        self.maxpool2 = nn.Sequential(
            nn.AvgPool1d(1, stride=1),
            nn.AvgPool1d(1, stride=1),
            nn.AvgPool1d(1, stride=1),
        )

        # Motif sizes
        self.motif_size_1 = 10
        self.motif_size_2 = 6

        # Reduction and classification layers
        self.reduction = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2, 2)))
        self.nn = nn.Sequential(nn.Linear(int(18 * 35 * self.max_m), num_classes))

        # Storage for gradient hooks (used in attribution)
        self._cam = []
        self._initial_out = []
        self._initial_pool = []

    def _hook_initial_q(self, grad):
        """Hook for capturing CNN motif gradients."""
        self._initial_out.append(grad)

    def _hook_initial_d(self, grad):
        """Hook for capturing pool gradients."""
        self._initial_pool.append(grad)

    def _hook_extract(self, grad):
        """Hook for capturing heatmap gradients."""
        self._cam.append(grad)

    def make_p(self, out, sax, seq_len, m):
        """
        Create motif-document interaction.

        Args:
            out: Projected input [batch, features, seq_len]
            sax: SAX encoded input [batch, sax_alpha, seq_len]
            seq_len: Sequence lengths
            m: Motif scale index

        Returns:
            Tuple of (CNN output, pooled SAX, interaction matrix)
        """
        cnn_net = getattr(self, f"cnn{m}")
        pool_net = getattr(self, f"maxpool{m}")
        ln_net = getattr(self, f"ln{m}")
        mo_size = getattr(self, f"motif_size_{m}")

        # CNN motif extraction
        q = cnn_net(out)  # [batch, d_out, reduced_len]
        q = torch.permute(ln_net(torch.permute(q, (0, 2, 1))), (0, 2, 1))

        store_q = q

        # SAX pooling
        d = pool_net(sax) * mo_size  # [batch, sax_alpha, reduced_len]
        d = d / seq_len[0]

        # Interaction matrix
        q = torch.matmul(d, q.permute(0, 2, 1))
        q = q.reshape((q.size(0), q.size(1), q.size(2), 1))

        return store_q, d, q

    def forward(self, x, sax, seq_len):
        """
        Forward pass for classification.

        Args:
            x: Input tensor [batch, seq_len, features]
            sax: SAX encoded tensor [batch, seq_len, sax_alpha]
            seq_len: Sequence lengths

        Returns:
            Classification logits [batch, num_classes]
        """
        out = self.proj(x)

        # Add positional encoding
        out = self.positional_encoding(out.permute(1, 0, 2), self.device)
        out = out.permute(1, 2, 0)

        sax = sax.permute(0, 2, 1)

        # Get motif-document interaction
        _, _, out_1 = self.make_p(out, sax, seq_len, 1)

        heat_map = out_1
        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.squeeze()
        heat_map = self.reduction(heat_map)

        # Classification
        heat_map = nn.Flatten()(heat_map)
        heat_map = self.nn(heat_map)

        return heat_map

    def forward_motif_importance(self, x, sax, seq_len):
        """
        Forward pass with gradient hooks for temporal attribution.

        Args:
            x: Input tensor [batch, seq_len, features]
            sax: SAX encoded tensor [batch, seq_len, sax_alpha]
            seq_len: Sequence lengths

        Returns:
            Tuple of (logits, cam gradients, initial_cam gradients, None)
        """
        # Reset gradient storage
        self._cam = []
        self._initial_out = []
        self._initial_pool = []

        self.Lin = x.size(1)
        self.seq_len = seq_len

        out = self.proj(x)
        out = self.positional_encoding(out.permute(1, 0, 2), self.device)
        out = out.permute(1, 2, 0)

        sax = sax.permute(0, 2, 1)

        # Get motif-document interaction with hooks
        self.out_1, self.pool_1, p_1 = self.make_p(out, sax, seq_len, 1)
        self.out_1.register_hook(self._hook_initial_q)

        heat_map = p_1
        heat_map.register_hook(self._hook_extract)

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.squeeze()
        heat_map = self.reduction(heat_map)

        heat_map = nn.Flatten()(heat_map)
        heat_map = self.nn(heat_map)

        return heat_map, self._cam, self._initial_out, None

    def assigning_importance(self, mo_level, kernel_size, unwrapped_len):
        """
        Assign motif-level importance back to original sequence positions.

        Args:
            mo_level: Motif-level importance [batch, reduced_len]
            kernel_size: Size of the convolution kernel
            unwrapped_len: Original sequence length

        Returns:
            Sequence-level importance [batch, unwrapped_len]
        """
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
    def calculate_motif_level(self, dp, m_i, initial_cam):
        """
        Calculate motif-level importance from gradients.

        Args:
            dp: Gradient tensor [batch, d_id, q_id]
            m_i: Motif index
            initial_cam: Initial CNN gradients

        Returns:
            Motif-level importance [batch, max_len]
        """
        d_comp = getattr(self, f"pool_{m_i}")
        q_comp = getattr(self, f"out_{m_i}")

        total = d_comp.size(0)
        max_len = d_comp.size(-1)
        q_id = q_comp.size(1)
        d_id = d_comp.size(1)

        all_motif_importance = torch.zeros((total, max_len)).to(self.device)
        temp = torch.zeros((total, d_comp.size(-1))).to(self.device)
        max_val_prot = torch.zeros((total, 1)).to(self.device)
        var_imp = torch.zeros((total, 1)).to(self.device)
        signs = torch.zeros((total, 1)).to(self.device)

        for i in range(q_id):
            for j in range(d_id):
                var_imp[:, :] = dp[:, j, i].unsqueeze(-1)
                signs[:, :] = torch.sign(var_imp)

                temp[:, :] = nn.ReLU()(signs * d_comp[:, j, :] * q_comp[:, i, :])

                max_val_prot[:, 0] = torch.max(temp, dim=1)[0].squeeze()
                temp[:, :] = (temp) / (max_val_prot + 1e-18)
                all_motif_importance += temp * abs(var_imp)

        return all_motif_importance


# Backward compatibility aliases
dataset = TimeSlicerDataset
timesliver_network = TimeSliverNetwork
