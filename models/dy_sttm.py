import torch
import torch.nn as nn

class SpatialTransformer(nn.Module):
    def __init__(self, input_dim, heads=4, dim_feedforward=512, dropout=0.1):
        super(SpatialTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.fc(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, heads=4, dim_feedforward=512, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.fc(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DynamicSTTM(nn.Module):
    def __init__(self, input_dim, num_spatial_layers=2, num_temporal_layers=2,
                 heads=4, dim_feedforward=512, dropout=0.1, num_classes=2):
        super(DynamicSTTM, self).__init__()
        
        self.spatial_transformer = nn.ModuleList(
            [SpatialTransformer(input_dim, heads, dim_feedforward, dropout)
             for _ in range(num_spatial_layers)]
        )
        
        self.temporal_transformer = nn.ModuleList(
            [TemporalTransformer(input_dim, heads, dim_feedforward, dropout)
             for _ in range(num_temporal_layers)]
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, spatial_dim]

        # Spatial Transformer
        spatial_out = x.permute(1, 0, 2)  # [seq_len, batch_size, spatial_dim]
        for layer in self.spatial_transformer:
            spatial_out = layer(spatial_out)

        # Temporal Transformer
        temporal_out = spatial_out.permute(1, 0, 2)  # [batch_size, seq_len, spatial_dim]
        temporal_out = temporal_out.permute(2, 0, 1)  # [spatial_dim, batch_size, seq_len]
        for layer in self.temporal_transformer:
            temporal_out = layer(temporal_out)

        temporal_out = temporal_out.permute(1, 2, 0)  # [batch_size, seq_len, spatial_dim]

        # Classification
        output = self.classifier(temporal_out)

        return output



