import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1

class CNNAttentionLSTM(nn.Module):
    def __init__(self, cnn_output_dim=1280, lstm_hidden=256, fc_hidden=64):
        super().__init__()

        # EfficientNet B1 backbone (7.79M params)
        base_model = efficientnet_b1(weights=None)
        checkpoint = torch.load('efficientnet_b1.pth')
        base_model.load_state_dict(checkpoint)
        self.cnn = base_model.features  # Output: [B*T, 1280, 7, 7]
        cnn_output_dim = 1280
        base_model = None
        
        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B*T, 1280, 1, 1] -> [B*T, 1280]
        
        # Lightweight transformer layer - reduced dimensions
        self.transformer = nn.TransformerEncoderLayer(
            d_model=cnn_output_dim,
            nhead=8,
            dim_feedforward=640,  # Reduced from 2560
            dropout=0.1,
            batch_first=True
        )
        
        # Smaller LSTM - 2 layers instead of 3
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden,
            num_layers=2,  # Reduced from 3
            batch_first=True,
            dropout=0.1
        )
        
        # Command and speed integration
        # Now using 6-class one-hot vector (indices 0-5)
        self.cmd_proj = nn.Linear(6, 32)  # One-hot command (6 classes)
        self.speed_proj = nn.Linear(1, 32)
        
        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + 32 + 32, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden, 3)  # [steer, gas, brake]
        )

    def forward(self, img_seq, cmd, speed):
        # img_seq: [B, T, C, H, W]
        # cmd: [B, 6] (one-hot encoded)
        # speed: [B, 1]
        
        B, T, C, H, W = img_seq.shape
        
        # CNN feature extraction
        img_seq = img_seq.view(B * T, C, H, W)  # [B*T, C, H, W]
        features = self.cnn(img_seq)  # [B*T, 1280, 7, 7]
        features = self.gap(features).squeeze(-1).squeeze(-1)  # [B*T, 1280]
        
        # Reshape for transformer
        features = features.view(B, T, -1)  # [B, T, 1280]
        
        # Transformer attention
        features = self.transformer(features)  # [B, T, 1280]
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # [B, T, 256]
        lstm_out = lstm_out[:, -1, :]  # Take last timestep [B, 256]
        
        # Process command and speed
        cmd_feat = self.cmd_proj(cmd)  # [B, 32]
        speed_feat = self.speed_proj(speed)  # [B, 32]
        
        # Combine features
        combined = torch.cat([lstm_out, cmd_feat, speed_feat], dim=1)  # [B, 320]
        
        # Final prediction
        output = self.fc(combined)  # [B, 3]
        
        return output
