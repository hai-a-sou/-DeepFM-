"""Memmap-backed PyTorch Dataset for the Criteo dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CriteoDataset(Dataset):
    """Dataset that reads from numpy memmap files for memory-efficient access."""

    def __init__(self, processed_dir, split="train"):
        self.processed_dir = Path(processed_dir)

        self.numerical = np.memmap(
            self.processed_dir / "numerical.npy", dtype=np.float32, mode="r",
            shape=self._get_shape("numerical.npy", 13)
        )
        self.categorical = np.memmap(
            self.processed_dir / "categorical.npy", dtype=np.int32, mode="r",
            shape=self._get_shape("categorical.npy", 26)
        )
        self.labels = np.memmap(
            self.processed_dir / "labels.npy", dtype=np.float32, mode="r"
        )

        indices_path = self.processed_dir / f"{split}_indices.npy"
        self.indices = np.load(indices_path).astype(np.int64)

    def _get_shape(self, filename, num_cols):
        """Infer the 2D shape from file size and column count."""
        total_bytes = (self.processed_dir / filename).stat().st_size
        rows = total_bytes // (num_cols * 4)  # float32/int32 = 4 bytes
        return (rows, num_cols)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        numerical = torch.tensor(self.numerical[real_idx], dtype=torch.float32)
        categorical = torch.tensor(self.categorical[real_idx], dtype=torch.long)
        label = torch.tensor(self.labels[real_idx], dtype=torch.float32)

        return numerical, categorical, label


def get_field_dims(processed_dir):
    """Load vocabulary sizes for each categorical field."""
    dims = np.load(Path(processed_dir) / "field_dims.npy")
    return dims.astype(np.int32).tolist()


def get_data_loaders(config):
    """Create train/val/test DataLoaders."""
    import platform

    num_workers = 4 if platform.system() != "Windows" else 0

    train_dataset = CriteoDataset(config.data.processed_dir, split="train")
    val_dataset = CriteoDataset(config.data.processed_dir, split="val")
    test_dataset = CriteoDataset(config.data.processed_dir, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    print(f"DataLoaders: train={len(train_dataset):,} val={len(val_dataset):,} test={len(test_dataset):,}")
    return train_loader, val_loader, test_loader
