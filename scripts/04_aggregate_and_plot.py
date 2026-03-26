from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gds.analysis.aggregate import aggregate_curves, collect_run_summaries, save_plots, save_summary
from gds.common.config import load_config


def _run_tsne(
    cfg: dict,
    artifacts_dir: Path,
    data_dir: Path,
    dataset_name: str,
    plot_dir: Path,
) -> None:
    """Generate t-SNE visualization of sample embeddings colored by difficulty.

    Works for both image datasets (pretrained ResNet-18 features) and text
    datasets (pretrained Qwen2-0.5B features).  Uses the first available
    scoring method's ranking to colour samples as easy / hard / random.
    """
    from gds.analysis.tsne_viz import (
        categorize_samples,
        compute_tsne,
        plot_tsne,
    )

    dataset_cfg = cfg["dataset"]
    scoring_cfg = cfg["scoring"]
    is_text = dataset_cfg.get("type", "image") == "text"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    batch_size = int(scoring_cfg.get("batch_size", 128))
    num_workers = int(scoring_cfg.get("num_workers", 2))

    # ---- Find a ranking file (prefer geometric methods, fall back to any) ----
    methods = scoring_cfg.get("methods", [])
    ranking_df = None
    used_method = None
    prefer_order = [
        "effective_rank", "corr_integral", "intrinsic_dimensionality_twonn",
        "forgetting_events", "perplexity_filtering", "semantic_dedup",
        "heuristic_filtering", "llm_classifier", "random",
    ]
    ordered = [m for m in prefer_order if m in methods] + [m for m in methods if m not in prefer_order]
    for m in ordered:
        rp = artifacts_dir / "rankings" / m / "scores.parquet"
        if rp.exists():
            ranking_df = pd.read_parquet(rp)
            used_method = m
            break
    if ranking_df is None:
        print("  No ranking files found; skipping t-SNE.")
        return
    print(f"  Using '{used_method}' ranking for t-SNE colouring.")

    # ---- Extract pretrained features ----
    if is_text:
        from gds.data.tiny_shakespeare import (
            TinyShakespeareDataset, build_text_loader,
            load_or_create_text_split,
        )
        from gds.scoring.pretrained_features import (
            extract_text_features, set_itos,
        )
        split_file = artifacts_dir / "splits" / f"tiny_shakespeare_split_seed{int(dataset_cfg['split_seed'])}.json"
        split, chunks, stoi, itos = load_or_create_text_split(
            data_dir, split_file,
            int(dataset_cfg.get("block_size", 256)),
            float(dataset_cfg.get("val_fraction", 0.1)),
            int(dataset_cfg["split_seed"]),
        )
        ds = TinyShakespeareDataset(chunks, split.train_ids)
        loader = build_text_loader(ds, batch_size, num_workers, shuffle=False)
        set_itos(itos)
        print("  Extracting pretrained Qwen2-0.5B features for t-SNE...")
        embeddings, labels, sample_ids_list = extract_text_features(loader, device)
        sample_ids = np.array(sample_ids_list)
        labels = np.array(labels)
    else:
        from gds.data.datasets import (
            build_indexed_dataset, build_loader,
            get_dataset_info, load_or_create_split, make_eval_transform,
        )
        from gds.scoring.pretrained_features import extract_image_features

        info = get_dataset_info(dataset_name)
        split_seed = int(dataset_cfg["split_seed"])
        val_size = int(dataset_cfg["val_size"])
        split_file = artifacts_dir / "splits" / f"{dataset_name}_split_seed{split_seed}.json"
        split = load_or_create_split(data_dir, split_file, dataset_name, val_size, split_seed)
        eval_tf = make_eval_transform(dataset_name)
        ds = build_indexed_dataset(data_dir, dataset_name, True, eval_tf, split.train_ids)
        loader = build_loader(ds, batch_size, num_workers, shuffle=False)
        print("  Extracting pretrained ResNet-18 features for t-SNE...")
        embeddings, labels_list, sample_ids_list = extract_image_features(loader, device)
        sample_ids = np.array(sample_ids_list)
        labels = np.array(labels_list)

    print(f"  Embeddings shape: {embeddings.shape}")

    # ---- Map ranking scores to sample indices ----
    print(f"  Mapping {len(ranking_df)} ranking scores to {len(sample_ids)} samples...")
    score_map = dict(zip(
        ranking_df["sample_id"].values,
        ranking_df["score"].values,
    ))
    scores = np.array([score_map.get(int(sid), 0) for sid in sample_ids], dtype=np.float32)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # ---- Categorize and run t-SNE ----
    tsne_samples = min(1000, len(sample_ids) // 4)
    categories = categorize_samples(
        sample_ids=sample_ids,
        scores=scores,
        n_per_category=tsne_samples,
    )

    all_idx = np.unique(np.concatenate(list(categories.values())))
    print(f"  Running t-SNE on {len(all_idx)} samples...")
    sub_embeddings = embeddings[all_idx]
    coords_2d = compute_tsne(sub_embeddings)

    idx_map = {old: new for new, old in enumerate(all_idx)}
    remapped_cats = {}
    for cat, indices in categories.items():
        remapped_cats[cat] = np.array([idx_map[i] for i in indices if i in idx_map])

    ds_label = dataset_name.replace("_", " ").title()
    method_label = used_method.replace("_", " ").title()
    plot_tsne(
        coords_2d=coords_2d,
        categories=remapped_cats,
        title=f"t-SNE Embeddings — {ds_label}\n(easy=low {method_label} score, hard=high {method_label} score)",
        out_path=plot_dir / "tsne_embeddings.png",
        dpi=int(cfg["analysis"].get("plot_dpi", 150)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run metrics and plot curves.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    dataset_name = cfg["dataset"].get("name", "mnist")

    run_df = collect_run_summaries(artifacts_dir / "training")
    if run_df.empty:
        raise RuntimeError("No run_summary.json files found. Train first.")

    summary_df = aggregate_curves(run_df)
    summary_dir = artifacts_dir / "summary"
    plot_dir = artifacts_dir / "plots"
    csv_path, json_path = save_summary(summary_df=summary_df, summary_dir=summary_dir)
    primary_metric = cfg.get("analysis", {}).get("primary_metric", "accuracy")
    generated = save_plots(
        summary_df=summary_df,
        plot_dir=plot_dir,
        dpi=int(cfg["analysis"]["plot_dpi"]),
        dataset_name=dataset_name,
        primary_metric=primary_metric,
        artifacts_dir=artifacts_dir,
    )

    print(f"\nSaved summary CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Generated {len(generated)} analysis artifacts.")

    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    try:
        _run_tsne(
            cfg=cfg,
            artifacts_dir=artifacts_dir,
            data_dir=data_dir,
            dataset_name=dataset_name,
            plot_dir=plot_dir,
        )
    except Exception as e:
        print(f"  t-SNE visualization failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
