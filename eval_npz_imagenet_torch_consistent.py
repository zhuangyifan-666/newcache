#!/usr/bin/env python3
"""
Self-consistent ImageNet-256 evaluation for .npz sample batches.

This script intentionally does NOT mix feature spaces:
all metrics are computed with the same torchvision InceptionV3 model.

Metrics:
- Inception Score (IS) on fake logits
- Fréchet Inception Distance (FID) on torchvision-Inception pool3 features
- Precision / Recall on the same feature space

Important:
- The FID from this script is self-consistent, but it is not directly
  comparable to ADM / TensorFlow evaluator numbers from papers.
- For tractability, Precision / Recall can be computed on a deterministic
  subset of features via --pr-samples.

Dependencies:
    pip install torch torchvision numpy scipy tqdm
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch import Tensor
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


DEFAULT_REF_NPZ = os.path.expanduser("~/.cache/adm_eval/VIRTUAL_imagenet256_labeled.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="One or more .npz files containing generated samples in arr_0.",
    )
    parser.add_argument(
        "--ref-npz",
        default=DEFAULT_REF_NPZ,
        help="Reference .npz containing arr_0 real images. "
        "Its mu/sigma fields are ignored on purpose.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", default="imagenet256_metrics_torch_consistent.json")
    parser.add_argument(
        "--real-cache",
        default=None,
        help="Optional cache file for real features/statistics. "
        "Default: <ref-npz>.torchvision_inception_cache.npz",
    )
    parser.add_argument(
        "--pr-samples",
        type=int,
        default=10000,
        help="Maximum number of real/fake features used for precision-recall.",
    )
    parser.add_argument("--nhood-size", type=int, default=5)
    parser.add_argument("--distance-row-batch", type=int, default=512)
    parser.add_argument("--distance-col-batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def build_inception(device: str):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()
    return model


def preprocess_batch(images: np.ndarray, device: str) -> Tensor:
    """
    images: [N, H, W, C] or [N, C, H, W], uint8 in [0, 255]
    returns: normalized float tensor [N, 3, 299, 299]
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape={images.shape}")

    if images.shape[-1] in (1, 3):
        images = images.transpose(0, 3, 1, 2)
    elif images.shape[1] not in (1, 3):
        raise ValueError(f"Unsupported image layout: {images.shape}")

    tensor = torch.from_numpy(images.astype(np.float32, copy=False)).to(device)
    tensor = tensor / 255.0
    tensor = F.interpolate(
        tensor,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def inception_features_and_logits(x: Tensor, model) -> Tuple[Tensor, Tensor]:
    # This follows torchvision InceptionV3 forward closely, while exposing pool3 features.
    x = model.Conv2d_1a_3x3(x)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    feat = torch.flatten(x, 1)
    feat = model.dropout(feat)
    logits = model.fc(feat)
    return feat, logits


def extract_features_and_logits(
    images_np: np.ndarray,
    model,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    logits = []

    with torch.no_grad():
        for start in tqdm(range(0, len(images_np), batch_size), desc="Extracting features"):
            batch_np = images_np[start : start + batch_size]
            batch_tensor = preprocess_batch(batch_np, device)
            feat, logit = inception_features_and_logits(batch_tensor, model)
            features.append(feat.cpu().numpy())
            logits.append(logit.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(logits, axis=0)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def calculate_inception_score(logits: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    probs = softmax(logits, axis=1)
    scores = []
    for idx in range(splits):
        start = idx * logits.shape[0] // splits
        end = (idx + 1) * logits.shape[0] // splits
        part = probs[start:end]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def calculate_fid(
    feat_fake: np.ndarray,
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu_fake = np.mean(feat_fake, axis=0)
    sigma_fake = np.cov(feat_fake, rowvar=False)

    diff = mu_fake - mu_real
    covmean, _ = linalg.sqrtm(sigma_fake @ sigma_real, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_fake.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma_fake + offset) @ (sigma_real + offset), disp=False)

    if np.iscomplexobj(covmean):
        imag_abs = np.max(np.abs(np.imag(covmean)))
        if imag_abs > 1e-3:
            print(f"[Warn] Large imaginary component in sqrtm: {imag_abs:.6e}")
        covmean = np.real(covmean)

    fid = diff.dot(diff) + np.trace(sigma_fake + sigma_real - 2.0 * covmean)
    return max(float(np.real(fid)), 0.0)


def maybe_cache_real_features(
    ref_npz_path: str,
    cache_path: str,
    model,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        print(f"[Loading Real Cache] {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        return cached["feat_real"], cached["mu_real"], cached["sigma_real"]

    print(f"[Loading Ref Images] {ref_npz_path}")
    ref_data = np.load(ref_npz_path, allow_pickle=False)
    if "arr_0" not in ref_data:
        raise KeyError(f"Reference npz missing arr_0. Keys: {list(ref_data.keys())}")

    real_imgs = ref_data["arr_0"]
    print("[Extracting Real Features]")
    feat_real, _ = extract_features_and_logits(real_imgs, model, device, batch_size)
    mu_real = np.mean(feat_real, axis=0)
    sigma_real = np.cov(feat_real, rowvar=False)

    np.savez(cache_path, feat_real=feat_real, mu_real=mu_real, sigma_real=sigma_real)
    print(f"[Saved Real Cache] {cache_path}")
    return feat_real, mu_real, sigma_real


def subsample_features(features: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or len(features) <= max_items:
        return features
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=max_items, replace=False)
    return features[indices]


def compute_kth_radii(
    features: np.ndarray,
    k: int,
    device: str,
    row_batch: int,
    col_batch: int,
) -> np.ndarray:
    num = len(features)
    radii = np.empty(num, dtype=np.float32)

    for row_start in tqdm(range(0, num, row_batch), desc="Estimating manifold radii"):
        row_end = min(row_start + row_batch, num)
        row = torch.from_numpy(features[row_start:row_end]).to(device=device, dtype=torch.float32)
        topk_values = None

        for col_start in range(0, num, col_batch):
            col_end = min(col_start + col_batch, num)
            col = torch.from_numpy(features[col_start:col_end]).to(device=device, dtype=torch.float32)
            dists = torch.cdist(row, col, p=2)

            overlap_start = max(row_start, col_start)
            overlap_end = min(row_end, col_end)
            if overlap_start < overlap_end:
                row_idx = torch.arange(overlap_start - row_start, overlap_end - row_start, device=device)
                col_idx = torch.arange(overlap_start - col_start, overlap_end - col_start, device=device)
                dists[row_idx, col_idx] = float("inf")

            if topk_values is None:
                topk_values = torch.topk(dists, k=k, dim=1, largest=False).values
            else:
                merged = torch.cat([topk_values, dists], dim=1)
                topk_values = torch.topk(merged, k=k, dim=1, largest=False).values

        radii[row_start:row_end] = topk_values[:, -1].cpu().numpy()

    return radii


def manifold_membership_ratio(
    query_features: np.ndarray,
    manifold_features: np.ndarray,
    manifold_radii: np.ndarray,
    device: str,
    row_batch: int,
    col_batch: int,
    desc: str,
) -> float:
    covered = np.zeros(len(query_features), dtype=bool)

    for row_start in tqdm(range(0, len(query_features), row_batch), desc=desc):
        row_end = min(row_start + row_batch, len(query_features))
        row = torch.from_numpy(query_features[row_start:row_end]).to(device=device, dtype=torch.float32)
        is_covered = torch.zeros(row.shape[0], dtype=torch.bool, device=device)

        for col_start in range(0, len(manifold_features), col_batch):
            col_end = min(col_start + col_batch, len(manifold_features))
            col = torch.from_numpy(manifold_features[col_start:col_end]).to(device=device, dtype=torch.float32)
            radii = torch.from_numpy(manifold_radii[col_start:col_end]).to(device=device, dtype=torch.float32)
            dists = torch.cdist(row, col, p=2)
            is_covered |= (dists <= radii.unsqueeze(0)).any(dim=1)
            if bool(is_covered.all()):
                break

        covered[row_start:row_end] = is_covered.cpu().numpy()

    return float(covered.mean())


def calculate_precision_recall(
    feat_real: np.ndarray,
    feat_fake: np.ndarray,
    nhood_size: int,
    device: str,
    row_batch: int,
    col_batch: int,
) -> Tuple[float, float]:
    radii_real = compute_kth_radii(feat_real, nhood_size, device, row_batch, col_batch)
    precision = manifold_membership_ratio(
        feat_fake,
        feat_real,
        radii_real,
        device,
        row_batch,
        col_batch,
        desc="Precision coverage",
    )

    radii_fake = compute_kth_radii(feat_fake, nhood_size, device, row_batch, col_batch)
    recall = manifold_membership_ratio(
        feat_real,
        feat_fake,
        radii_fake,
        device,
        row_batch,
        col_batch,
        desc="Recall coverage",
    )
    return precision, recall


def load_npz_images(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    if "arr_0" not in data:
        raise KeyError(f"NPZ missing arr_0. Keys: {list(data.keys())}")
    return data["arr_0"]


def main() -> None:
    args = parse_args()
    cache_path = args.real_cache or f"{args.ref_npz}.torchvision_inception_cache.npz"

    print(f"[Device] {args.device}")
    print("[Loading Model] torchvision InceptionV3")
    model = build_inception(args.device)

    feat_real, mu_real, sigma_real = maybe_cache_real_features(
        ref_npz_path=args.ref_npz,
        cache_path=cache_path,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = []
    for sample_path in args.samples:
        print(f"\n[Evaluating] {sample_path}")
        if not os.path.exists(sample_path):
            print(f"[Warn] Missing sample file: {sample_path}")
            continue

        fake_imgs = load_npz_images(sample_path)
        print(f"Data shape: {fake_imgs.shape}, dtype: {fake_imgs.dtype}")

        feat_fake, logits_fake = extract_features_and_logits(
            fake_imgs,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
        )

        is_mean, is_std = calculate_inception_score(logits_fake)
        fid = calculate_fid(feat_fake, mu_real, sigma_real)

        feat_real_pr = subsample_features(feat_real, args.pr_samples, args.seed)
        feat_fake_pr = subsample_features(feat_fake, args.pr_samples, args.seed + 1)
        precision, recall = calculate_precision_recall(
            feat_real_pr,
            feat_fake_pr,
            nhood_size=args.nhood_size,
            device=args.device,
            row_batch=args.distance_row_batch,
            col_batch=args.distance_col_batch,
        )

        result = {
            "sample": sample_path,
            "sample_count": int(len(fake_imgs)),
            "feature_space": "torchvision_inception_v3_pool3",
            "fid_note": "Self-consistent torch-Inception FID computed against ref arr_0. "
            "Not directly comparable to ADM/TF FID.",
            "real_feature_count_for_fid": int(len(feat_real)),
            "real_feature_count_for_pr": int(len(feat_real_pr)),
            "fake_feature_count_for_pr": int(len(feat_fake_pr)),
            "inception_score_mean": float(is_mean),
            "inception_score_std": float(is_std),
            "fid": float(fid),
            "precision": float(precision),
            "recall": float(recall),
        }
        results.append(result)

        print("-" * 30)
        print(f"IS: {is_mean:.4f} +/- {is_std:.4f}")
        print(f"FID: {fid:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("-" * 30)

    with open(args.json_out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Saved] {args.json_out}")


if __name__ == "__main__":
    main()
