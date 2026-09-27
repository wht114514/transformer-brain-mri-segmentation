import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TransBTS(nn.Module):
    # 3D CNN 编码器 + Transformer 瓶颈 + 3D CNN 解码器
    def __init__(self, in_channels=4, num_classes=3, embed_dim=512,
                 num_heads=8, num_layers=4, depths=(32, 64, 128, 256)):
        super().__init__()
        self.num_classes = num_classes

        self.stem = ConvBlock(in_channels, depths[0])

        self.enc1 = EncoderStage(depths[0], depths[1])
        self.enc2 = EncoderStage(depths[1], depths[2])
        self.enc3 = EncoderStage(depths[2], depths[3])

        self.proj = nn.Linear(depths[3], embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.unproj = nn.Linear(embed_dim, depths[3])

        self.dec1 = DecoderStage(depths[3], depths[2], depths[2])
        self.dec2 = DecoderStage(depths[2], depths[1], depths[1])
        self.dec3 = DecoderStage(depths[1], depths[0], depths[0])

        self.head = nn.Conv3d(depths[0], num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b, c, d, h, w = s3.shape
        n = d * h * w

        tokens = s3.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        tokens = tokens + self.pos_embed[:, :n, :]
        tokens = self.transformer(tokens)

        tokens = self.unproj(tokens)
        feat = tokens.transpose(1, 2).view(b, c, d, h, w)

        x = self.dec1(feat, s2)
        x = self.dec2(x, s1)
        x = self.dec3(x, s0)

        return self.head(x)
