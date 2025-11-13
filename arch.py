import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([x, h_cur], dim=1)  # (B, C+H, H, W)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        H, W = spatial_size
        return (torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
                torch.zeros(batch_size, self.hidden_dim, H, W, device=device))


class ConvLSTM(nn.Module):
    def __init__(self, input_channels=23, hidden_dim=32):  # Increased hidden_dim
        super().__init__()

        # Encoder with reduced downsampling (only 2 pooling layers)
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # 256 -> 128
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # 128 -> 64
            # Removed the third MaxPool3d layer to preserve spatial resolution
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        # ConvLSTM cell (spatiotemporal modeling) with bigger hidden size
        self.convlstm = ConvLSTMCell(input_dim=64, hidden_dim=hidden_dim)

        # Decoder upsampling back to 256x256 from 64 spatial size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128 -> 256
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),  # final fire mask
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Encoder expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.encoder(x)  # (B, 64, T, 64, 64)
        _, C_enc, T_enc, H_enc, W_enc = x.shape

        # ConvLSTM over timesteps
        h, c = self.convlstm.init_hidden(B, (H_enc, W_enc), x.device)
        for t in range(T_enc):
            h, c = self.convlstm(x[:, :, t], (h, c))

        # Decode last hidden state
        out = self.decoder(h)
        return out
