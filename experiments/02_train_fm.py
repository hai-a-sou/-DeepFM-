"""Step 2: Train Factorization Machine (FM) baseline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from deepfm_ctr.config import load_config
from deepfm_ctr.data.dataset import get_field_dims, get_data_loaders
from deepfm_ctr.models.fm import FactorizationMachineModel
from deepfm_ctr.training.trainer import Trainer
from deepfm_ctr.training.evaluator import evaluate_all
from deepfm_ctr.utils.reproducibility import seed_everything, get_device, count_parameters
from deepfm_ctr.visualization.plotter import plot_training_curves, plot_prediction_distribution


def main():
    config = load_config("configs/default.yaml")
    seed_everything(config.data.random_seed)
    device = get_device()
    print(f"Device: {device}")

    # Data
    field_dims = get_field_dims(config.data.processed_dir)
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Model
    model = FactorizationMachineModel(
        field_dims=field_dims, num_numerical=13,
        embed_dim=config.model.embed_dim,
    )
    print(f"FM parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader, model_name="fm")

    # Evaluate on test set
    labels, scores = trainer.predict(test_loader)
    metrics = evaluate_all(labels, scores)
    print(f"\nTest results: AUC={metrics['auc']:.4f} LogLoss={metrics['logloss']:.4f}")

    # Save metrics
    with open("results/fm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_training_curves(history, "results/figures/fm_training_curves.png")
    plot_prediction_distribution(labels, scores, "FM", "results/figures/fm_pred_dist.png")


if __name__ == "__main__":
    main()
