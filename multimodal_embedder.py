"""
ProjectD Multimodal Embedder
Extracts normalized dense representations from text and images,
projecting both modalities into a shared perceptual subspace (32-dim).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional
import urllib.request

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class SharedPerceptualProjector(nn.Module):
    """
    Projects raw 512-dim CLIP text/vision embeddings into
    ProjectD's 32-dim cross-network manifold.
    """
    def __init__(self, input_dim: int = 512, target_dim: int = 32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return nn.functional.normalize(projected, p=2, dim=-1)


class MultimodalEmbedder:
    def __init__(self, target_dim: int = 32):
        self.target_dim = target_dim
        self.model_id = "openai/clip-vit-base-patch32"
        print(f"[*] Loading Multimodal Vision/Text Encoder ({self.model_id})...")
        
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.model.eval()

        self.projector = SharedPerceptualProjector(input_dim=512, target_dim=target_dim)
        self.projector.eval()

    def _fetch_image(self, image_source: str) -> Optional[Image.Image]:
        try:
            if image_source.startswith("http://") or image_source.startswith("https://"):
                req = urllib.request.Request(
                    image_source,
                    headers={"User-Agent": "ProjectD-Ingestion/1.0"}
                )
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    raw_data = resp.read()
                return Image.open(io.BytesIO(raw_data)).convert("RGB")
            elif Path(image_source).exists():
                return Image.open(image_source).convert("RGB")
        except Exception as e:
            print(f"[!] Warning: Failed to load image ({image_source}): {e}")
        return None

    def embed_candidate(self, text: Optional[str] = None, image_url: Optional[str] = None) -> np.ndarray:
        """
        Embeds a candidate item. If both text and image are present,
        it fuses them via late-fusion projection.
        """
        text_emb = None
        img_emb = None

        with torch.no_grad():
            if text and text.strip():
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                t_features = self.model.get_text_features(**inputs)
                text_emb = nn.functional.normalize(t_features, p=2, dim=-1)

            if image_url and image_url.strip():
                img = self._fetch_image(image_url)
                if img is not None:
                    inputs = self.processor(images=img, return_tensors="pt")
                    i_features = self.model.get_image_features(**inputs)
                    img_emb = nn.functional.normalize(i_features, p=2, dim=-1)

            # Modality Fusion
            if text_emb is not None and img_emb is not None:
                # 50/50 semantic and visual aesthetic fusion
                fused = 0.5 * text_emb + 0.5 * img_emb
                fused = nn.functional.normalize(fused, p=2, dim=-1)
                projected = self.projector(fused)
            elif text_emb is not None:
                projected = self.projector(text_emb)
            elif img_emb is not None:
                projected = self.projector(img_emb)
            else:
                # Fallback zero-vector if no content provided
                return np.zeros(self.target_dim, dtype=np.float32)

        return projected.squeeze(0).cpu().numpy()