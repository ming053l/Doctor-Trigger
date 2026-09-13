"""FaceSigns-style fractal watermark reimplementation.

Trimmed to the definitions the peer comparison uses."""

from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def logistic_keystream(length: int, seed: int, r: float = 3.99, warmup: int = 64) -> np.ndarray:
    # Generate chaotic keystream bits using logistic map.
    x = ((seed % 1_000_003) + 1) / 1_000_004.0
    bits = np.zeros(length, dtype=np.float32)
    idx = 0
    for i in range(length + warmup):
        x = r * x * (1.0 - x)
        if i >= warmup:
            bits[idx] = 1.0 if x > 0.5 else 0.0
            idx += 1
    return bits

def xor_encrypt_bits(message_bits: np.ndarray, key_seed: int) -> np.ndarray:
    ks = logistic_keystream(int(message_bits.size), key_seed)
    return np.bitwise_xor(message_bits.astype(np.uint8), ks.astype(np.uint8)).astype(np.float32)

def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def _d2xy(n: int, d: int) -> Tuple[int, int]:
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def hilbert_curve_points(order: int) -> np.ndarray:
    n = 2 ** order
    pts = np.zeros((n * n, 2), dtype=np.int32)
    for d in range(n * n):
        x, y = _d2xy(n, d)
        pts[d] = [x, y]
    return pts

def generate_fractal_map(height: int, width: int, key_seed: int, order: int = 6) -> np.ndarray:
    # Build a multi-scale Hilbert-based fractal watermark map.
    rng = np.random.default_rng(key_seed)
    n = 2 ** order
    base = np.zeros((n, n), dtype=np.float32)
    pts = hilbert_curve_points(order)
    for i in range(len(pts) - 1):
        x1, y1 = int(pts[i, 0]), int(pts[i, 1])
        x2, y2 = int(pts[i + 1, 0]), int(pts[i + 1, 1])
        cv2.line(base, (x1, y1), (x2, y2), color=1.0, thickness=1)

    base = cv2.GaussianBlur(base, (3, 3), 0)
    layers = []
    for scale in (1.0, 0.5, 0.25):
        resized = cv2.resize(base, (max(1, int(n * scale)), max(1, int(n * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((n, n), dtype=np.float32)
        h2, w2 = resized.shape
        ox = int(rng.integers(0, max(1, n - w2 + 1)))
        oy = int(rng.integers(0, max(1, n - h2 + 1)))
        canvas[oy : oy + h2, ox : ox + w2] = resized
        angle = float(rng.uniform(0.0, 360.0))
        m = cv2.getRotationMatrix2D((n / 2.0, n / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(canvas, m, (n, n), flags=cv2.INTER_LINEAR)
        weight = float(rng.uniform(0.6, 1.2))
        layers.append(weight * rotated)

    fractal = np.sum(layers, axis=0)
    fractal = fractal - fractal.min()
    denom = fractal.max() + 1e-8
    fractal = fractal / denom
    fractal = cv2.resize(fractal, (width, height), interpolation=cv2.INTER_LINEAR)
    return fractal.astype(np.float32)

def generate_fractal_batch(keys: Sequence[int], h: int, w: int, device: torch.device) -> torch.Tensor:
    maps = [generate_fractal_map(h, w, int(k)) for k in keys]
    arr = np.stack(maps, axis=0)
    ten = torch.from_numpy(arr).unsqueeze(1).to(device=device, dtype=torch.float32)
    return ten

def make_message_bits(keys: Sequence[int], msg_len: int) -> np.ndarray:
    msgs = []
    for k in keys:
        rng = np.random.default_rng(int(k) & 0xFFFFFFFF)
        msg = rng.integers(0, 2, size=msg_len, dtype=np.uint8)
        msgs.append(msg)
    return np.stack(msgs, axis=0)

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class SEResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return F.relu(x + y, inplace=True)

class FractalEncoder(nn.Module):
    def __init__(self, image_size: int = 128, msg_len: int = 128, channels: int = 32):
        super().__init__()
        self.image_size = int(image_size)
        self.msg_len = int(msg_len)

        self.img_stem = nn.Sequential(
            ConvBlock(3, channels),
            SEResBlock(channels),
            SEResBlock(channels),
        )

        patch = max(8, self.image_size // 8)
        self.msg_proj = nn.Sequential(
            nn.Linear(msg_len, patch * patch),
            nn.ReLU(inplace=True),
        )
        self.wm_path = nn.Sequential(
            ConvBlock(2, channels // 2),
            SEResBlock(channels // 2),
        )

        self.fuse = nn.Sequential(
            ConvBlock(channels + channels // 2, channels),
            SEResBlock(channels),
            SEResBlock(channels),
        )
        self.to_delta = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, encrypted_bits: torch.Tensor, fractal_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = image.shape
        feat_img = self.img_stem(image)

        patch = max(8, self.image_size // 8)
        msg = self.msg_proj(encrypted_bits).view(b, 1, patch, patch)
        msg = F.interpolate(msg, size=(h, w), mode="bilinear", align_corners=False)

        wm_feat = self.wm_path(torch.cat([msg, fractal_map], dim=1))
        fused = self.fuse(torch.cat([feat_img, wm_feat], dim=1))
        delta = torch.tanh(self.to_delta(fused)) * 0.12
        watermarked = (image + delta).clamp(0.0, 1.0)
        return watermarked, delta

class FractalDecoder(nn.Module):
    def __init__(self, msg_len: int = 128, channels: int = 32):
        super().__init__()
        self.feat = nn.Sequential(
            ConvBlock(3, channels),
            SEResBlock(channels),
            ConvBlock(channels, channels * 2, stride=2),
            SEResBlock(channels * 2),
            ConvBlock(channels * 2, channels * 2, stride=2),
            SEResBlock(channels * 2),
        )
        self.msg_head = nn.Linear(channels * 2, msg_len)
        self.map_head = nn.Sequential(
            ConvBlock(channels * 2, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.feat(image)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        msg_logits = self.msg_head(pooled)
        map_logits = self.map_head(feat)
        map_logits = F.interpolate(map_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return msg_logits, map_logits

def bit_error_rate(logits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5).float()
    ber = torch.mean(torch.abs(pred - target_bits), dim=1)
    return ber
