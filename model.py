import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CNNAttentionLSTM(nn.Module):
    def __init__(self, cnn_output_dim=1280, lstm_hidden=128, fc_hidden=64):
        super().__init__()

        # EfficientNet B0 como extractor visual
        base_model = efficientnet_b0(weights=None)  # usa weights=EfficientNet_B0_Weights.DEFAULT si deseas pretrained
        self.cnn = base_model.features  # Salida: [B*T, 1280, 7, 7]

        self.pool = nn.AdaptiveAvgPool2d(1)  # Reduce a [B*T, 1280, 1, 1]

        # Proyección de velocidad: escalar → vector
        self.speed_fc = nn.Linear(1, cnn_output_dim)

        # Atención: peso escalar por frame
        self.attn = nn.MultiheadAttention(embed_dim=cnn_output_dim, num_heads=1, batch_first=True)

        # LSTM
        self.lstm = nn.LSTM(input_size=cnn_output_dim, hidden_size=lstm_hidden, batch_first=True)

        # MLP final
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + 4, fc_hidden),  # velocidad ya está fusionada
            nn.ReLU(),
            nn.Linear(fc_hidden, 3),
            nn.Tanh()
        )

    def forward(self, img_seq, cmd, speed):
        B, T, C, H, W = img_seq.shape
        x = img_seq.view(B*T, C, H, W)
        x = self.cnn(x)  # (B*T, 1280, H', W')
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B*T, 1280)
        x = x.view(B, T, -1)  # (B, T, 1280)

        # Expandir velocidad y concatenar a cada timestep
        speed_feat = self.speed_fc(speed).unsqueeze(1).repeat(1, T, 1)  # (B, T, 1280)
        x = x + speed_feat  # fusión

        # Atención temporal
        x_attn, _ = self.attn(x, x, x)  # (B, T, 1280)

        # LSTM temporal
        lstm_out, _ = self.lstm(x_attn)
        last_out = lstm_out[:, -1, :]  # (B, hidden)

        # Concat con comando (one-hot)
        full_input = torch.cat([last_out, cmd], dim=1)
        return self.fc(full_input)
