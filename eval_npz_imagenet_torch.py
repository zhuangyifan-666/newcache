#!/usr/bin/env python3
"""
Evaluate class-to-image .npz sample batches on ImageNet-256 using PyTorch.
Replaces OpenAI's TF1 evaluator.

Computes:
- Inception Score (IS)
- Fréchet Inception Distance (FID)
- Precision & Recall (using Manifold Estimation)

Dependencies:
    pip install torch torchvision numpy scipy tqdm
"""
import argparse
import json
import os
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from tqdm import tqdm

# Default paths
DEFAULT_SAMPLES = [
    "/mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/universal_pix_workdirs/exp_PixelGen_XL/val_ode50_cfg1.0/predict/output.npz",
]
DEFAULT_REF_NPZ = os.path.expanduser("~/.cache/adm_eval/VIRTUAL_imagenet256_labeled.npz")


def get_inception_model(device):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()
    return model, weights.transforms()


def preprocess_batch(images, transform, device):
    """
    images: [N, H, W, C] or [N, C, H, W], numpy array, 0-255 uint8
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D array, got {images.shape}")

    # Handle channel order
    if images.shape[-1] in [1, 3]:
        # NHWC -> NCHW
        images = images.transpose(0, 3, 1, 2)
    
    # Normalize to [0.0, 1.0]
    images = images.astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(images).to(device)
    
    # Apply standard Inception transforms (resize to 299x299, normalize)
    # Note: The transform expects PIL Image range [0,1], which we have.
    # We apply it manually to batches.
    tensor = F.interpolate(tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize using ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor


def get_features_and_logits(images_np, model, transform, device, batch_size=32):
    features = []
    logits = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(images_np), batch_size), desc="Extracting features"):
            batch_np = images_np[i:i+batch_size]
            batch_tensor = preprocess_batch(batch_np, transform, device)
            
            # Forward pass
            # We need to hijack the pool3 layer. Let's do it manually.
            x = batch_tensor
            # N x 3 x 299 x 299
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
            # N x 2048 x 8 x 8
            x = F.adaptive_avg_pool2d(x, (1, 1))
            # N x 2048 x 1 x 1
            feat = torch.flatten(x, 1) # N x 2048
            
            # Calculate logits
            logit = model.fc(feat)
            
            features.append(feat.cpu().numpy())
            logits.append(logit.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(logits, axis=0)


# ------------------------------
# Metrics Calculations
# ------------------------------

def calculate_is(logits, splits=10):
    """Inception Score"""
    scores = []
    for i in range(splits):
        part = logits[(i * logits.shape[0] // splits):((i + 1) * logits.shape[0] // splits), :]
        py = np.mean(softmax(part, axis=1), axis=0)
        scores.append(np.exp(np.mean(np.sum(softmax(part, axis=1) * (np.log(softmax(part, axis=1)) - np.log(py)[None, :]), axis=1))))
    return np.mean(scores), np.std(scores)

def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def calculate_fid(feat_fake, mu_real, sigma_real):
    """Fréchet Inception Distance"""
    mu_fake = np.mean(feat_fake, axis=0)
    sigma_fake = np.cov(feat_fake, rowvar=False)
    
    m = np.square(mu_fake - mu_real).sum()
    s, _ = linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
    fid = m + np.trace(sigma_fake + sigma_real - 2 * s)
    return float(np.real(fid))

def compute_pairwise_distances(X, Y):
    """Compute pairwise Euclidean distances."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.sqrt(dists)

def calculate_precision_recall(feat_real, feat_fake, nhood_size=5):
    """
    Precision and Recall for Generative Models (Kynkaanniemi et al., 2019)
    """
    # Manifold estimation: find nearest neighbor distances in real data
    real_dists = compute_pairwise_distances(feat_real, feat_real)
    np.fill_diagonal(real_dists, np.inf)
    radii_real = np.sort(real_dists, axis=1)[:, nhood_size - 1]
    
    # Precision: % of fake points inside real manifold
    fake_to_real = compute_pairwise_distances(feat_fake, feat_real)
    precision = float(np.mean(np.any(fake_to_real <= radii_real[None, :], axis=1)))
    
    # Recall: % of real points covered by fake manifold
    # Estimate fake manifold
    fake_dists = compute_pairwise_distances(feat_fake, feat_fake)
    np.fill_diagonal(fake_dists, np.inf)
    radii_fake = np.sort(fake_dists, axis=1)[:, nhood_size - 1]
    
    real_to_fake = compute_pairwise_distances(feat_real, feat_fake)
    recall = float(np.mean(np.any(real_to_fake <= radii_fake[None, :], axis=1)))
    
    return precision, recall


def load_npz_images(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    if "arr_0" not in data:
        raise KeyError(f"NPZ missing 'arr_0'. Keys: {list(data.keys())}")
    return data["arr_0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--ref-npz", default=DEFAULT_REF_NPZ, help="Path to VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", default="imagenet256_metrics_torch.json")
    args = parser.parse_args()

    print(f"[Device] {args.device}")
    
    # 1. Load Reference Data
    print(f"[Loading Ref] {args.ref_npz}")
    ref_data = np.load(args.ref_npz, allow_pickle=False)
    mu_real = ref_data['mu']
    sigma_real = ref_data['sigma']
    real_imgs = ref_data['arr_0'] # 10k real images for Precision/Recall
    
    # 2. Load Model
    print("[Loading Model] InceptionV3")
    model, preprocess = get_inception_model(args.device)
    
    # 3. Extract Real Features (for P&R)
    print("[Extracting Real Features]")
    feat_real, _ = get_features_and_logits(real_imgs, model, preprocess, args.device, args.batch_size)
    
    # 4. Evaluate each sample
    results = []
    for sample_path in args.samples:
        print(f"\n[Evaluating] {sample_path}")
        if not os.path.exists(sample_path):
            print(f"[Warn] Missing: {sample_path}")
            continue
            
        fake_imgs = load_npz_images(sample_path)
        print(f"Data shape: {fake_imgs.shape}")
        
        # Shuffle (like original script)
        np.random.seed(2026)
        np.random.shuffle(fake_imgs)
        
        # Extract
        feat_fake, logits_fake = get_features_and_logits(fake_imgs, model, preprocess, args.device, args.batch_size)
        
        # Calc metrics
        is_mean, is_std = calculate_is(logits_fake)
        fid = calculate_fid(feat_fake, mu_real, sigma_real)
        precision, recall = calculate_precision_recall(feat_real, feat_fake)
        
        res = {
            "sample": sample_path,
            "inception_score": float(is_mean),
            "fid": float(fid),
            "precision": float(precision),
            "recall": float(recall)
        }
        results.append(res)
        
        # Print
        print("-" * 30)
        print(f"IS: {is_mean:.4f}")
        print(f"FID: {fid:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("-" * 30)

    # Save JSON
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {args.json_out}")

if __name__ == "__main__":
    main()
    