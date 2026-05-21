"""Training loop with early stopping, checkpointing, and LR scheduling."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from .evaluator import evaluate_auc, evaluate_logloss


class Trainer:
    """Model trainer with early stopping and checkpointing."""

    def __init__(self, model, config, device, checkpoint_dir="results/checkpoints"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6,
        )

        # Early stopping state
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.best_epoch = 0
        self.history = []

    def _collect_predictions(self, loader):
        """Run model on an entire loader, collect labels and scores."""
        self.model.eval()
        all_labels = []
        all_scores = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for numerical, categorical, labels in loader:
                numerical = numerical.to(self.device)
                categorical = categorical.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(numerical, categorical)
                loss = self.criterion(logits.squeeze(), labels)

                scores = torch.sigmoid(logits.squeeze())
                all_labels.append(labels.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
                total_loss += loss.item()
                n_batches += 1

        return (
            np.concatenate(all_labels),
            np.concatenate(all_scores),
            total_loss / max(n_batches, 1),
        )

    def train_epoch(self, train_loader):
        """Train one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for numerical, categorical, labels in train_loader:
            numerical = numerical.to(self.device)
            categorical = categorical.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(numerical, categorical).squeeze()
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, loader):
        """Evaluate AUC and log loss on a given loader."""
        labels, scores, avg_loss = self._collect_predictions(loader)
        auc = evaluate_auc(labels, scores)
        logloss = evaluate_logloss(labels, scores)
        return auc, logloss, avg_loss

    def train(self, train_loader, val_loader, model_name="model"):
        """Full training loop with early stopping."""
        max_epochs = self.config.training.max_epochs
        patience = self.config.training.early_stopping_patience

        print(f"\nTraining {model_name} (max {max_epochs} epochs, patience={patience})")
        print("-" * 60)

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_auc, val_logloss, val_loss = self.evaluate(val_loader)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_auc": val_auc, "val_logloss": val_logloss, "lr": lr,
            })

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | lr={lr:.1e}")

            # Early stopping logic
            if val_auc > self.best_val_metric:
                self.best_val_metric = val_auc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(model_name, epoch, val_auc, is_best=True)
                print(f"  -> New best AUC {val_auc:.4f}, checkpoint saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (best val_auc={self.best_val_metric:.4f} at epoch {self.best_epoch})")
                    break

            self.scheduler.step(val_auc)

        # Restore best model
        self._load_best(model_name)
        return self.history

    def _save_checkpoint(self, model_name, epoch, val_auc, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auc": val_auc,
            "history": self.history,
        }
        # Always save latest
        torch.save(checkpoint, self.checkpoint_dir / f"{model_name}_latest.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / f"{model_name}_best.pth")

    def _load_best(self, model_name):
        best_path = self.checkpoint_dir / f"{model_name}_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Restored best model from epoch {checkpoint['epoch']} (AUC={checkpoint['val_auc']:.4f})")

    def predict(self, loader):
        """Return predicted scores for a data loader."""
        labels, scores, _ = self._collect_predictions(loader)
        return labels, scores
