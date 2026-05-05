"""
Panel Retrieval Score.
Uses a frozen non-geolocation encoder (DINOv2) to measure whether
generated images retrieve the correct reference panel over hard negatives.
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional
from pathlib import Path

import config


class PanelRetriever:
    """Retrieve matching reference panels using DINOv2 embeddings."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None

    def load_model(self):
        """Load DINOv2 model."""
        print(f"Loading DINOv2 ({self.model_name})...")
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.model.to(self.device)
        self.model.eval()

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("DINOv2 loaded.")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single image to embedding."""
        if self.model is None:
            self.load_model()

        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model(img_tensor)
        return features.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Encode a batch of images."""
        if self.model is None:
            self.load_model()

        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            features = self.model(tensors)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def panel_embedding(self, images: list[Image.Image]) -> np.ndarray:
        """Compute mean embedding for a panel of images."""
        embeddings = self.encode_batch(images)
        return embeddings.mean(axis=0)


class PanelRetrievalScore:
    """Compute Panel Retrieval Score.

    For each generated image, check whether it retrieves the correct
    reference panel over hard-negative panels.
    """

    def __init__(self):
        self.retriever = PanelRetriever()

    def compute_score(self, gen_images: list[Image.Image],
                      target_panel: list[Image.Image],
                      negative_panels: list[list[Image.Image]]) -> dict:
        """Compute retrieval score.

        Args:
            gen_images: generated images for the target place
            target_panel: real reference images for the target place
            negative_panels: list of reference panels for hard negatives

        Returns:
            Dict with retrieval metrics
        """
        # Encode all panels
        target_emb = self.retriever.panel_embedding(target_panel)
        neg_embs = [self.retriever.panel_embedding(panel) for panel in negative_panels]
        all_panel_embs = np.stack([target_emb] + neg_embs)  # [N_panels, D]

        # Encode generated images
        gen_embs = self.retriever.encode_batch(gen_images)  # [N_gen, D]

        # Compute cosine similarity
        # Normalize
        all_panel_embs_norm = all_panel_embs / (np.linalg.norm(all_panel_embs, axis=1, keepdims=True) + 1e-8)
        gen_embs_norm = gen_embs / (np.linalg.norm(gen_embs, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix: [N_gen, N_panels]
        sim_matrix = gen_embs_norm @ all_panel_embs_norm.T

        # Metrics
        # 1. Retrieval accuracy: how often is target panel ranked first?
        ranks = sim_matrix.argsort(axis=1)[:, ::-1]  # descending
        target_ranks = np.array([np.where(ranks[i] == 0)[0][0] for i in range(len(gen_embs))])

        retrieval_acc = float((target_ranks == 0).mean())
        mean_rank = float(target_ranks.mean())
        mrr = float((1.0 / (target_ranks + 1)).mean())

        # 2. Mean similarity to target vs negatives
        sim_to_target = float(sim_matrix[:, 0].mean())
        sim_to_negatives = float(sim_matrix[:, 1:].mean())
        sim_gap = sim_to_target - sim_to_negatives

        return {
            "retrieval_acc": retrieval_acc,      # higher = better
            "mean_reciprocal_rank": mrr,          # higher = better
            "mean_rank": mean_rank,               # lower = better
            "sim_to_target": sim_to_target,
            "sim_to_negatives": sim_to_negatives,
            "sim_gap": sim_gap,                   # higher = better
            "per_image_ranks": target_ranks.tolist(),
        }


class RoundTripGeolocator:
    """Round-trip geolocation using GeoCLIP (appendix metric).

    GPS -> Generate -> GeoCLIP predict GPS -> measure distance error.
    """

    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self.model = None

    def load_model(self):
        """Load GeoCLIP model."""
        try:
            from geoclip import GeoCLIP
            print("Loading GeoCLIP...")
            self.model = GeoCLIP()
            self.model.to(self.device)
            print("GeoCLIP loaded.")
        except ImportError:
            print("WARNING: geoclip not installed. Round-trip metric unavailable.")
            print("Install with: pip install geoclip")
            self.model = None

    @torch.no_grad()
    def predict_location(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """Predict GPS location from image."""
        if self.model is None:
            self.load_model()
        if self.model is None:
            return []

        top_pred_gps, top_pred_prob = self.model.predict(image, top_k=top_k)
        predictions = []
        for gps, prob in zip(top_pred_gps.cpu().numpy(), top_pred_prob.cpu().numpy()):
            predictions.append({
                "lat": float(gps[0]),
                "lon": float(gps[1]),
                "prob": float(prob),
            })
        return predictions

    def compute_round_trip_error(self, image: Image.Image,
                                 target_lat: float, target_lon: float) -> dict:
        """Compute round-trip distance error."""
        from geopy.distance import geodesic

        preds = self.predict_location(image, top_k=5)
        if not preds:
            return {"error_km": float("inf"), "predictions": []}

        best_pred = preds[0]
        error_km = geodesic(
            (target_lat, target_lon),
            (best_pred["lat"], best_pred["lon"])
        ).km

        return {
            "error_km": error_km,
            "predictions": preds,
        }


def main():
    """Quick test."""
    print("Panel Retrieval Score module loaded successfully.")
    print("To run: instantiate PanelRetrievalScore and call compute_score()")


if __name__ == "__main__":
    main()
