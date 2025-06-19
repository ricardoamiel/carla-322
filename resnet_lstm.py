import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, cnn_output_dim=512, lstm_hidden=128, fc_hidden=64):
        super().__init__()
        # Backbone CNN (ResNet18)
        resnet = models.resnet18(pretrained=False) # True for pretrained weights
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Output shape: (B, 512, 1, 1)

        # LSTM for temporal context
        self.lstm = nn.LSTM(input_size=cnn_output_dim, hidden_size=lstm_hidden, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + 4 + 1, fc_hidden),  # Add high-level command (4) + speed (1)
            nn.ReLU(),
            nn.Linear(fc_hidden, 3),  # Steer, Gas, Brake
            nn.Tanh()  # Optional: output in [-1, 1] for Steer
        )

    def forward(self, img_seq, cmd, speed):
        B, T, C, H, W = img_seq.shape # B batch, T timestep, C chan, H y W
        img_seq = img_seq.view(B*T, C, H, W)
        features = self.cnn(img_seq).squeeze(-1).squeeze(-1)  # Shape: [B*T, 512]
        features = features.view(B, T, -1)
        
        lstm_out, _ = self.lstm(features)  # [B, T, hidden]
        last_out = lstm_out[:, -1, :]  # use last output
        
        full_input = torch.cat([last_out, cmd, speed], dim=1)
        return self.fc(full_input)
