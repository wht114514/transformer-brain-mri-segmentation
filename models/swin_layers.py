import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).view(b, d, h, w)


class PatchMerge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.merge = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.merge(x)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).view(b, d, h, w)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def window_partition(self, x, window_size):
        b, d, h, w = x.shape
        x = x.view(b, d, h // window_size, window_size,
                   w // window_size, window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, window_size * window_size, d)

    def window_reverse(self, x, window_size, h, w):
        b = int(x.shape[0] / ((h // window_size) * (w // window_size)))
        x = x.view(b, h // window_size, w // window_size,
                   window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, -1, h, w)

    def forward(self, x):
        b, d, h, w = x.shape
        shortcut = x

        ws = min(self.attn.window_size, h, w)
        x = self.window_partition(x, ws)
        x = x.transpose(1, 2).contiguous()
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.attn(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.window_reverse(x, ws, h, w)

        return x + shortcut
