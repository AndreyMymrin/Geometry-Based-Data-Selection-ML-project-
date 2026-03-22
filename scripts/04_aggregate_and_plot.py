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
from gds.scoring.registry import is_forgetting_method


def _run_tsne(
    cfg: dict,
    artifacts_dir: Path,
    data_dir: Path,
    dataset_name: str,
    plot_dir: Path,
) -> None:
    """Generate t-SNE visualization of sample embeddings colored by difficulty."""
    from gds.analysis.tsne_viz import (
        categorize_samples,
        compute_tsne,
        extract_embeddings,
        plot_tsne,
    )
    from gds.data.datasets import (
        build_indexed_dataset,
        build_loader,
        get_dataset_info,
        load_or_create_split,
        make_eval_transform,
        make_train_transform,
    )
    from gds.models.simple_cnn import SimpleCNN

    dataset_cfg = cfg["dataset"]
    scoring_cfg = cfg["scoring"]
    info = get_dataset_info(dataset_name)

    # Find the forgetting-based method to load its ranking scores
    methods = scoring_cfg.get("methods", [])
    forgetting_method = None
    for m in methods:
        if is_forgetting_method(m):
            forgetting_method = m
            break
    if forgetting_method is None:
        print("  No forgetting-based method found; skipping t-SNE.")
        return

    ranking_path = artifacts_dir / "rankings" / forgetting_method / "scores.parquet"
    if not ranking_path.exists():
        print(f"  Ranking file not found: {ranking_path}; skipping t-SNE.")
        return
    ranking_df = pd.read_parquet(ranking_path)

    # Load or create split (image datasets only)
    if dataset_cfg.get("type") == "text":
        print("  t-SNE not supported for text datasets; skipping.")
        return
    split_seed = int(dataset_cfg["split_seed"])
    val_size = int(dataset_cfg["val_size"])
    split_file = artifacts_dir / "splits" / f"{dataset_name}_split_seed{split_seed}.json"
    split = load_or_create_split(
        data_dir=data_dir,
        split_file=split_file,
        dataset_name=dataset_name,
        val_size=val_size,
        seed=split_seed,
    )

    batch_size = int(scoring_cfg.get("batch_size", 128))
    num_workers = int(scoring_cfg.get("num_workers", 2))

    # Build eval dataset for clean embeddings
    eval_tf = make_eval_transform(dataset_name)
    ds = build_indexed_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train=True,
        transform=eval_tf,
        indices=split.train_ids,
    )
    loader = build_loader(ds, batch_size, num_workers, shuffle=False)

    # Build training dataset for quick model training
    train_tf = make_train_transform(dataset_name)
    train_ds = build_indexed_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train=True,
        transform=train_tf,
        indices=split.train_ids,
    )
    train_loader = build_loader(train_ds, batch_size, num_workers, shuffle=True)

    device = torch.device("cpu")
    model = SimpleCNN(num_classes=info["num_classes"], in_channels=info["in_channels"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = int(scoring_cfg.get("num_epochs", 10))
    print(f"  Training SimpleCNN for {num_epochs} epochs (CPU) for embeddings...")
    model.train()
    for epoch in range(num_epochs):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Extract embeddings
    print("  Extracting embeddings...")
    embeddings, labels, sample_ids = extract_embeddings(model, loader, device)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Map ranking scores to sample indices
    score_map = dict(zip(
        ranking_df["sample_id"].values,
        ranking_df["score"].values,
    ))
    scores = np.array([score_map.get(sid, 0) for sid in sample_ids], dtype=np.float32)

    # Categorize samples
    tsne_samples = min(1000, len(sample_ids) // 4)
    categories = categorize_samples(
        sample_ids=sample_ids,
        scores=scores,
        n_per_category=tsne_samples,
    )

    # Subsample for t-SNE speed
    all_idx = np.unique(np.concatenate(list(categories.values())))
    print(f"  Running t-SNE on {len(all_idx)} samples...")
    sub_embeddings = embeddings[all_idx]
    coords_2d = compute_tsne(sub_embeddings)

    # Remap category indices to subsampled array
    idx_map = {old: new for new, old in enumerate(all_idx)}
    remapped_cats = {}
    for cat, indices in categories.items():
        remapped_cats[cat] = np.array([idx_map[i] for i in indices if i in idx_map])

    ds_label = dataset_name.replace("_", " ").title()
    plot_tsne(
        coords_2d=coords_2d,
        categories=remapped_cats,
        title=f"t-SNE Embeddings - {ds_label}\n(easy=low forgetting, hard=high forgetting)",
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
