import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class CNNAttentionLSTM(nn.Module):
    def __init__(self, cnn_output_dim=1280, lstm_hidden=512, fc_hidden=64, pretrained=True): # 1280. 512 y 64
        super().__init__()

        # EfficientNet B0 como extractor visual
        base_model = efficientnet_b0(weights=None)
        checkpoint = torch.load("efficientnet_b0_rwightman-7f5810bc.pth")
        base_model.load_state_dict(checkpoint)
        self.cnn = base_model.features  # Salida: [B*T, 1280, H', W']

        self.pool = nn.AdaptiveAvgPool2d(1)  # Reduce a [B*T, 1280, 1, 1]

        #self.cnn_proj = nn.Linear(1280, cnn_output_dim) # si el cnn output dim es 1280 no hace nada

        # Proyección de velocidad: escalar → vector
        self.speed_fusion = nn.Sequential(
            nn.Linear(cnn_output_dim + 1, cnn_output_dim),
            nn.ReLU(),
            nn.Linear(cnn_output_dim, cnn_output_dim)
        )

        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cnn_output_dim,
                nhead=8,
                dim_feedforward=cnn_output_dim*4,
                batch_first=True
            ),
            num_layers=2
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )

        # MLP final
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + 64, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, fc_hidden//2),
            nn.ReLU(),
            nn.Linear(fc_hidden//2, 3),
            nn.Tanh()
        )

        self.norm1 = nn.LayerNorm(cnn_output_dim)
        self.norm2 = nn.LayerNorm(cnn_output_dim)

        self.cmd_embed = nn.Embedding(4, 16)
        self.cmd_fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU()
        )

    def forward(self, img_seq, cmd, speed):
        B, T, C, H, W = img_seq.shape
        x = img_seq.view(B*T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).flatten(1)
        #x = self.cnn_proj(x)
        x = x.view(B, T, -1)

        speed_expanded = speed.unsqueeze(1).repeat(1, T, 1)
        visual_speed = torch.cat([x, speed_expanded], dim=-1)
        x = self.speed_fusion(visual_speed)
        x = self.norm1(x)

        x = self.attn(x)
        x = self.norm2(x)

        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]

        cmd_embedded = self.cmd_embed(cmd)
        cmd_embedded = self.cmd_fc(cmd_embedded)

        combined = torch.cat([last_out, cmd_embedded], dim=1)
        return self.fc(combined)
