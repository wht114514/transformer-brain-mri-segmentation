import torch
import torch.nn as nn

from .swin_layers import SwinBlock, PatchEmbed, PatchMerge


class USwinTransformer(nn.Module):
    # Unet + Swin Transformer 结构
    def __init__(self, in_channels=1, num_classes=3, embed_dim=96,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=7):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(in_channels, embed_dim)
        self.pos_drop = nn.Dropout(0.1)

        self.layers1 = self.make_layer(embed_dim, embed_dim, depths[0], num_heads[0], window_size)
        self.layers2 = self.make_layer(embed_dim, embed_dim * 2, depths[1], num_heads[1], window_size)
        self.layers3 = self.make_layer(embed_dim * 2, embed_dim * 4, depths[2], num_heads[2], window_size)
        self.layers4 = self.make_layer(embed_dim * 4, embed_dim * 8, depths[3], num_heads[3], window_size)

        self.merge1 = PatchMerge(embed_dim, embed_dim * 2)
        self.merge2 = PatchMerge(embed_dim * 2, embed_dim * 4)
        self.merge3 = PatchMerge(embed_dim * 4, embed_dim * 8)

        dim = embed_dim * 8
        self.up3 = self.up_block(dim, embed_dim * 4)
        self.up2 = self.up_block(embed_dim * 4, embed_dim * 2)
        self.up1 = self.up_block(embed_dim * 2, embed_dim)
        self.up0 = self.up_block(embed_dim, embed_dim // 2)

        self.head = nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)

    def make_layer(self, in_dim, out_dim, depth, num_heads, window_size):
        blocks = []
        if in_dim != out_dim:
            blocks.append(PatchMerge(in_dim, out_dim))
        for _ in range(depth):
            blocks.append(SwinBlock(out_dim, num_heads, window_size))
        return nn.Sequential(*blocks)

    def up_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        s1 = self.layers1(x)
        s2 = self.layers2(s1)
        s3 = self.layers3(s2)
        s4 = self.layers4(s3)

        x = self.up3(s4) + s3
        x = self.up2(x) + s2
        x = self.up1(x) + s1
        x = self.up0(x)

        return self.head(x)
