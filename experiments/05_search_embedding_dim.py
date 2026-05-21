"""Step 5: Grid search over embedding dimensions for DeepFM."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from deepfm_ctr.config import load_config
from deepfm_ctr.data.dataset import get_field_dims, get_data_loaders
from deepfm_ctr.models.deepfm import DeepFM
from deepfm_ctr.training.trainer import Trainer
from deepfm_ctr.utils.reproducibility import seed_everything, get_device, count_parameters
from deepfm_ctr.visualization.plotter import plot_embedding_search


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Device: {device}")

    # Data (shared across runs)
    field_dims = get_field_dims(config.data.processed_dir)
    train_loader, val_loader, test_loader = get_data_loaders(config)

    embed_dims = config.search.embed_dims
    results = {}

    for embed_dim in embed_dims:
        seed_everything(config.data.random_seed)

        print(f"\n{'='*60}")
        print(f"Training DeepFM with embed_dim={embed_dim}")
        print(f"{'='*60}")

        model = DeepFM(
            field_dims=field_dims, num_numerical=13,
            embed_dim=embed_dim,
            mlp_dims=config.model.mlp_dims,
            dropout=config.model.dropout,
            use_batch_norm=config.model.use_batch_norm,
        ).to(device)

        print(f"Parameters: {count_parameters(model):,}")

        trainer = Trainer(model, config, device)
        history = trainer.train(train_loader, val_loader, model_name=f"deepfm_emb{embed_dim}")

        # Record best validation AUC from training history
        best_val_auc = max(h["val_auc"] for h in history)
        results[embed_dim] = best_val_auc
        print(f"embed_dim={embed_dim}: best_val_auc={best_val_auc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("Embedding Dimension Search Results")
    print(f"{'='*60}")
    for dim, auc in sorted(results.items()):
        print(f"  embed_dim={dim:3d}: AUC={auc:.4f}")

    best_dim = max(results, key=results.get)
    print(f"\nBest embedding dimension: {best_dim} (AUC={results[best_dim]:.4f})")

    # Save and plot
    with open("results/embedding_search.json", "w") as f:
        json.dump({"results": results, "best_dim": best_dim}, f, indent=2)

    plot_embedding_search(results, "results/figures/embedding_dim_search.png")


if __name__ == "__main__":
    main()
