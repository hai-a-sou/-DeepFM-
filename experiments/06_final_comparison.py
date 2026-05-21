"""Step 6: Aggregate results and produce final comparison figures."""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from deepfm_ctr.config import load_config
from deepfm_ctr.data.dataset import get_field_dims, get_data_loaders
from deepfm_ctr.models.fm import FactorizationMachineModel
from deepfm_ctr.models.deepfm import DeepFM
from deepfm_ctr.training.trainer import Trainer
from deepfm_ctr.utils.reproducibility import seed_everything, get_device
from deepfm_ctr.visualization.plotter import plot_roc_curves, plot_metric_comparison

def load_model_and_predict(model_class, checkpoint_name, config, field_dims, test_loader, device, **kwargs):
    """Load a trained model from checkpoint and get test predictions."""
    model = model_class(field_dims=field_dims, num_numerical=13, **kwargs).to(device)

    checkpoint_path = f"results/checkpoints/{checkpoint_name}_best.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {checkpoint_name} from epoch {checkpoint['epoch']} (val_auc={checkpoint['val_auc']:.4f})")
    else:
        print(f"Warning: {checkpoint_path} not found. Using untrained model.")

    trainer = Trainer(model, config, device)
    return trainer.predict(test_loader)


def main():
    config = load_config("configs/default.yaml")
    seed_everything(config.data.random_seed)
    device = get_device()
    print(f"Device: {device}")

    # Data
    field_dims = get_field_dims(config.data.processed_dir)
    train_loader, val_loader, test_loader = get_data_loaders(config)

    labels_dict = {}
    scores_dict = {}
    all_metrics = {}

    # FM
    print("\n--- FM ---")
    if Path("results/fm_metrics.json").exists():
        with open("results/fm_metrics.json") as f:
            all_metrics["FM"] = json.load(f)
        fm_labels, fm_scores = load_model_and_predict(
            FactorizationMachineModel, "fm", config, field_dims, test_loader, device,
            embed_dim=config.model.embed_dim,
        )
        labels_dict["FM"] = fm_labels
        scores_dict["FM"] = fm_scores
        print(f"FM: AUC={all_metrics['FM']['auc']:.4f} LogLoss={all_metrics['FM']['logloss']:.4f}")
    else:
        print("FM metrics not found, skipping.")

    # DeepFM
    print("\n--- DeepFM ---")
    if Path("results/deepfm_metrics.json").exists():
        with open("results/deepfm_metrics.json") as f:
            all_metrics["DeepFM"] = json.load(f)
        dfm_labels, dfm_scores = load_model_and_predict(
            DeepFM, "deepfm", config, field_dims, test_loader, device,
            embed_dim=config.model.embed_dim,
            mlp_dims=config.model.mlp_dims,
            dropout=config.model.dropout,
            use_batch_norm=config.model.use_batch_norm,
        )
        labels_dict["DeepFM"] = dfm_labels
        scores_dict["DeepFM"] = dfm_scores
        print(f"DeepFM: AUC={all_metrics['DeepFM']['auc']:.4f} LogLoss={all_metrics['DeepFM']['logloss']:.4f}")
    else:
        print("DeepFM metrics not found, skipping.")

    # XGBoost
    print("\n--- XGBoost ---")
    xgb_pkl_path = Path("results/checkpoints/xgboost_best.pkl")
    if Path("results/xgboost_metrics.json").exists():
        with open("results/xgboost_metrics.json") as f:
            all_metrics["XGBoost"] = json.load(f)
        print(f"XGBoost: AUC={all_metrics['XGBoost']['auc']:.4f} LogLoss={all_metrics['XGBoost']['logloss']:.4f}")

        if xgb_pkl_path.exists():
            import pickle as pkl
            import xgboost as xgb
            with open(xgb_pkl_path, "rb") as f:
                saved = pkl.load(f)
            xgb_model = saved["model"]
            y_test = saved["test_labels"]

            # Load test features efficiently via memmap
            indices = np.load(config.data.processed_dir / "test_indices.npy").astype(np.int64)
            sorted_idx = np.sort(indices)
            n_rows = int((config.data.processed_dir / "numerical.npy").stat().st_size / (13 * 4))
            numer = np.array(np.memmap(
                config.data.processed_dir / "numerical.npy", dtype=np.float32, mode="r", shape=(n_rows, 13)
            )[sorted_idx])
            categ_shape = (n_rows, 26)
            categ = np.array(np.memmap(
                config.data.processed_dir / "categorical.npy", dtype=np.int32, mode="r", shape=categ_shape
            )[sorted_idx])
            X_test = np.concatenate([numer, categ.astype(np.float32)], axis=1)

            xgb_scores = xgb_model.predict(xgb.DMatrix(X_test))
            labels_dict["XGBoost"] = y_test
            scores_dict["XGBoost"] = xgb_scores
            print(f"  XGBoost loaded for ROC plot")
        else:
            print("  (XGBoost checkpoint not found, ROC unavailable)")
    else:
        print("XGBoost metrics not found, skipping.")

    # Generate comparison figures
    print("\n--- Generating figures ---")
    if len(scores_dict) >= 2:
        plot_roc_curves(labels_dict, scores_dict, "results/figures/roc_comparison.png")
        print("  ROC comparison saved to results/figures/roc_comparison.png")

    if len(all_metrics) >= 2:
        plot_metric_comparison(all_metrics, "results/figures/model_comparison.png")
        print("  Model comparison saved to results/figures/model_comparison.png")

    # Print summary table
    print(f"\n{'='*60}")
    print("Final Results Summary")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'AUC':>8} {'LogLoss':>10} {'Accuracy':>10} {'F1':>8}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<12} {m['auc']:>8.4f} {m['logloss']:>10.4f} {m['accuracy']:>10.4f} {m['f1']:>8.4f}")

    # Also print embedding search results if available
    if Path("results/embedding_search.json").exists():
        with open("results/embedding_search.json") as f:
            es = json.load(f)
        print(f"\nBest embedding dim from search: {es['best_dim']} (AUC={es['results'][str(es['best_dim'])]:.4f})")


if __name__ == "__main__":
    main()
