import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7

class CNNAttentionLSTM(nn.Module):
    def __init__(self, cnn_output_dim=2560, lstm_hidden=512, fc_hidden=64):
        super().__init__()

        # EfficientNet B7 como extractor visual
        base_model = efficientnet_b7(weights=None)
        checkpoint = torch.load("efficientnet_b7_lukemelas-c5b4e57e.pth")
        base_model.load_state_dict(checkpoint)
        self.cnn = base_model.features  # Salida: [B*T, 2560, 7, 7]
        base_model = None

        self.pool = nn.AdaptiveAvgPool2d(1)  # Reduce a [B*T, 2560, 1, 1]

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
            hidden_size=512,  
            num_layers=3,     
            batch_first=True,
            dropout=0.2      
        )

        # MLP final
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + 64, fc_hidden),  # velocidad ya está fusionada
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, fc_hidden//2),
            nn.ReLU(),
            nn.Linear(fc_hidden//2, 3),
            nn.Tanh()
        )
        
        # Add throughout the network
        self.norm1 = nn.LayerNorm(cnn_output_dim)
        self.norm2 = nn.LayerNorm(cnn_output_dim)

        # Expand command processing
        self.cmd_embed = nn.Embedding(4, 16)  # Learn command embeddings
        self.cmd_fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU()
        )

    def forward(self, img_seq, cmd, speed):
        B, T, C, H, W = img_seq.shape
        x = img_seq.view(B*T, C, H, W)
        x = self.cnn(x)  # (B*T, 2560, H', W')
        x = self.pool(x).flatten(1)
        x = x.view(B, T, -1)  # (B, T, 2560)

        # Expandir velocidad y concatenar a cada timestep
        speed_expanded = speed.unsqueeze(1).repeat(1, T, 1)  # [B, T, 1]
        visual_speed = torch.cat([x, speed_expanded], dim=-1)
        x = self.speed_fusion(visual_speed)
        x = self.norm1(x)

        # Atención temporal
        x = self.attn(x)
        x = self.norm2(x)

        # LSTM temporal
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (B, hidden)

        cmd_embedded = self.cmd_embed(cmd)  # [B, 16]
        cmd_embedded = self.cmd_fc(cmd_embedded)  # [B, 64]

        # Prediction
        combined = torch.cat([last_out, cmd_embedded], dim=1)
        return self.fc(combined)
