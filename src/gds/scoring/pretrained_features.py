"""Extract hidden states / features from pretrained models.

Text:  Qwen2-0.5B  (HuggingFace transformers)
Image: ResNet-18    (torchvision, ImageNet weights)

Following the methodology of Yusupov et al. (2025), "From Internal
Representations to Text Quality: A Geometric Approach to LLM Evaluation",
geometric metrics are computed on hidden states of pretrained models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# ------------------------------------------------------------------ #
# Image: pretrained ResNet-18 hidden states
# ------------------------------------------------------------------ #

def extract_image_hidden_states(
    loader,
    device: torch.device,
) -> tuple[list[list[np.ndarray]], list[int], list[int]]:
    """Extract per-sample hidden states from pretrained ResNet-18.

    Each intermediate feature map (C, H, W) is reshaped to (H*W, C)
    so that rows = spatial positions, columns = channels.  This gives
    a point cloud of spatial locations in channel-feature space.

    Layers extracted: layer1, layer2, layer3, layer4 (4 layers).

    Returns
    -------
    all_hidden : list of N lists, each containing 4 np.ndarrays
    all_labels : list of N ints
    all_sids   : list of N ints
    """
    from torchvision import models

    print("  Loading pretrained ResNet-18 (ImageNet weights)...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.to(device)
    resnet.eval()

    # Hook into the 4 residual blocks
    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    activations: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook(module, input, output):
            activations[name] = output
        return hook

    hooks = []
    for name in layer_names:
        h = getattr(resnet, name).register_forward_hook(_make_hook(name))
        hooks.append(h)

    all_hidden, all_labels, all_sids = [], [], []

    # ResNet-18 expects 3 channels.  For 1-channel (MNIST) we repeat.
    # We resize small images to at least 64×64 (not the full 224×224) to
    # keep things fast while still getting meaningful multi-scale features.
    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc="  ResNet-18 features"):
            x = x.to(device)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            # Resize small images to 64×64 for reasonable feature maps
            h, w = x.shape[2], x.shape[3]
            if h < 64 or w < 64:
                x = torch.nn.functional.interpolate(
                    x, size=(96, 96), mode="bilinear", align_corners=False,
                )

            resnet(x)

            B = y.shape[0]
            for i in range(B):
                sample_hs = []
                for name in layer_names:
                    feat = activations[name][i]  # (C, H, W)
                    C, H, W = feat.shape
                    # Reshape to (H*W, C): spatial positions as points
                    mat = feat.permute(1, 2, 0).reshape(H * W, C).cpu().numpy()
                    sample_hs.append(mat)
                all_hidden.append(sample_hs)
            all_labels.extend(y.numpy().astype(int).tolist())
            all_sids.extend(sid.numpy().astype(int).tolist())

    for h in hooks:
        h.remove()

    print(f"  {len(all_hidden)} samples × {len(layer_names)} layers")
    for i, name in enumerate(layer_names):
        shape = all_hidden[0][i].shape
        print(f"    {name}: ({shape[0]}, {shape[1]})")
    return all_hidden, all_labels, all_sids


def extract_image_features(
    loader,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Extract flat features from pretrained ResNet-18 (avgpool layer).

    Returns features of shape (N, 512) — the penultimate layer.
    Used by intrinsic_dimensionality and semantic_dedup.
    """
    from torchvision import models

    print("  Loading pretrained ResNet-18 (ImageNet weights)...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.to(device)
    resnet.eval()

    # Remove the final classification layer
    features_list: list[torch.Tensor] = []

    def _hook(module, input, output):
        features_list.append(output.squeeze(-1).squeeze(-1))

    hook = resnet.avgpool.register_forward_hook(_hook)

    all_labels, all_sids = [], []
    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc="  ResNet-18 flat features"):
            x = x.to(device)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            h, w = x.shape[2], x.shape[3]
            if h < 64 or w < 64:
                x = torch.nn.functional.interpolate(
                    x, size=(96, 96), mode="bilinear", align_corners=False,
                )
            resnet(x)
            all_labels.extend(y.numpy().astype(int).tolist())
            all_sids.extend(sid.numpy().astype(int).tolist())

    hook.remove()

    features = torch.cat(features_list, dim=0).cpu().numpy()
    print(f"  Features shape: {features.shape}")
    return features, all_labels, all_sids


# ------------------------------------------------------------------ #
# Text: pretrained Qwen2-0.5B hidden states
# ------------------------------------------------------------------ #

def extract_text_hidden_states(
    loader,
    device: torch.device,
    model_name: str = "Qwen/Qwen2-0.5B",
) -> tuple[list[list[np.ndarray]], list[int], list[int]]:
    """Extract per-sample hidden states from pretrained Qwen2-0.5B.

    Each sample's character-level chunk is decoded to text, re-tokenised
    with Qwen2's tokeniser, and fed through the model.  Hidden states
    from all transformer layers are collected.

    Each hidden state has shape (n_tokens, d_hidden).

    Returns
    -------
    all_hidden : list of N lists, each containing L np.ndarrays
    all_labels : list of N ints
    all_sids   : list of N ints
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading pretrained {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()

    # We need the itos mapping to decode character-level tokens back to text.
    # The loader's dataset should have it, or we reconstruct from the data.
    # Since our TinyShakespeare chunks are character indices, we need itos.
    # We'll get it from the first batch and the dataset.

    all_hidden, all_labels, all_sids = [], [], []
    max_qwen_tokens = 512  # cap to avoid OOM

    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc=f"  {model_name} hidden states"):
            B = x.shape[0]
            for i in range(B):
                # x[i] is character-level token indices; decode to text
                # We need the mapping — for now, just use chr() since
                # TinyShakespeare uses a small char vocab
                char_ids = x[i].numpy().astype(int)
                # Attempt to decode (works for ASCII-range chars)
                text = _decode_char_ids(char_ids)

                # Tokenise with Qwen2 tokeniser
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_qwen_tokens,
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"].to(device)

                # Forward pass
                outputs = model(input_ids)
                # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, d_hidden)
                hidden_states = outputs.hidden_states

                # Skip embedding layer (index 0), take transformer layers
                sample_hs = []
                for layer_hs in hidden_states[1:]:
                    mat = layer_hs[0].float().cpu().numpy()  # (seq_len, d_hidden)
                    sample_hs.append(mat)
                all_hidden.append(sample_hs)

            # Labels and sample IDs
            all_labels.extend(y[:, 0].numpy().astype(int).tolist())
            if hasattr(sid, "numpy"):
                all_sids.extend(sid.numpy().astype(int).tolist())
            else:
                all_sids.extend([int(s) for s in sid])

    n_layers = len(all_hidden[0]) if all_hidden else 0
    print(f"  {len(all_hidden)} samples × {n_layers} layers")
    if all_hidden:
        print(f"    Layer shape example: {all_hidden[0][0].shape}")
    return all_hidden, all_labels, all_sids


def extract_text_features(
    loader,
    device: torch.device,
    model_name: str = "Qwen/Qwen2-0.5B",
) -> tuple[np.ndarray, list[int], list[int]]:
    """Extract flat features from pretrained Qwen2-0.5B (last hidden state, mean-pooled).

    Returns features of shape (N, d_hidden).
    Used by intrinsic_dimensionality and semantic_dedup.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading pretrained {model_name} for flat features...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()

    all_features, all_labels, all_sids = [], [], []
    max_qwen_tokens = 512

    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc=f"  {model_name} flat features"):
            B = x.shape[0]
            for i in range(B):
                char_ids = x[i].numpy().astype(int)
                text = _decode_char_ids(char_ids)

                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_qwen_tokens,
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"].to(device)

                outputs = model(input_ids)
                # Last hidden state: (1, seq_len, d_hidden) → mean pool → (d_hidden,)
                last_hs = outputs.hidden_states[-1]
                feat = last_hs[0].float().mean(dim=0).cpu().numpy()
                all_features.append(feat)

            all_labels.extend(y[:, 0].numpy().astype(int).tolist())
            if hasattr(sid, "numpy"):
                all_sids.extend(sid.numpy().astype(int).tolist())
            else:
                all_sids.extend([int(s) for s in sid])

    features = np.stack(all_features, axis=0)
    print(f"  Features shape: {features.shape}")
    return features, all_labels, all_sids


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

# Global itos mapping — set by the pipeline before calling text extractors
_ITOS: dict[int, str] | None = None


def set_itos(itos: dict[int, str]) -> None:
    """Set the character-level index-to-string mapping for text decoding."""
    global _ITOS
    _ITOS = itos


def _decode_char_ids(char_ids: np.ndarray) -> str:
    """Decode character-level token IDs back to text."""
    global _ITOS
    if _ITOS is not None:
        return "".join(_ITOS.get(int(c), "?") for c in char_ids)
    # Fallback: assume ASCII
    return "".join(chr(max(32, min(126, int(c)))) for c in char_ids)
