"""Streaming two-pass Criteo data preprocessing with memmap output.

The Criteo dataset has 39 tab-separated columns:
  col 0:  Label (0/1 click)
  col 1-13:  Numerical features I1-I13 (int/float, may be missing)
  col 14-39: Categorical features C1-C26 (hex-encoded hashes)

Pass 1: stream in chunks → compute numerical stats (Welford), count cat frequencies.
Pass 2: re-stream → transform, write to numpy memmap files.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

COLUMN_NAMES = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
NUMERICAL_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]


class WelfordAccumulator:
    """Online mean/variance computation (single-pass, numerically stable)."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        """Update with a 1D array of non-missing values."""
        for x in values:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self):
        return self.m2 / max(self.count - 1, 1)

    @property
    def std(self):
        return np.sqrt(self.variance)


class CriteoPreprocessor:
    """Streaming preprocessor for the Criteo display advertising dataset."""

    def __init__(self, config):
        self.config = config
        self.raw_path = Path(config.data.raw_path)
        self.processed_dir = Path(config.data.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Accumulators built during fit()
        self.numerical_stats = {col: WelfordAccumulator() for col in NUMERICAL_COLS}
        self.categorical_counts = {col: defaultdict(int) for col in CATEGORICAL_COLS}
        self.categorical_mappings = {}
        self.field_dims = []

        self.num_numerical = len(NUMERICAL_COLS)
        self.num_categorical = len(CATEGORICAL_COLS)

    def _count_rows(self):
        """Count total rows in raw file (fast)."""
        count = 0
        with open(self.raw_path, "r") as f:
            for _ in f:
                count += 1
        return count

    def fit(self):
        """Pass 1: stream data, accumulate numerical stats and categorical frequencies."""
        min_freq = self.config.preprocessing.categorical.min_frequency
        total_rows = self._count_rows()
        chunk_size = self.config.data.chunk_size
        n_chunks = (total_rows + chunk_size - 1) // chunk_size

        print(f"Pass 1 (fit): streaming {total_rows:,} rows in {n_chunks} chunks...")

        for chunk in tqdm(
            pd.read_csv(
                self.raw_path,
                sep="\t",
                header=None,
                names=COLUMN_NAMES,
                chunksize=chunk_size,
                dtype={col: str for col in CATEGORICAL_COLS},
            ),
            total=n_chunks,
        ):
            for col in NUMERICAL_COLS:
                series = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
                # clip to non-negative before log1p
                self.numerical_stats[col].update(np.clip(series.values, 0, None))

            for col in CATEGORICAL_COLS:
                for val in chunk[col].dropna().values:
                    self.categorical_counts[col][str(val)] += 1

        # Build categorical mappings: frequent values get their own index
        max_vocab = self.config.preprocessing.categorical.max_vocab_size
        for col in CATEGORICAL_COLS:
            # Sort by frequency descending, keep only min_frequency+ values
            freq_items = sorted(self.categorical_counts[col].items(), key=lambda x: -x[1])
            freq_items = [(v, c) for v, c in freq_items if c >= min_freq]

            if max_vocab and len(freq_items) > max_vocab:
                freq_items = freq_items[:max_vocab]

            # Index 0 = UNK (unknown / rare values)
            mapping = {val: idx + 1 for idx, (val, _) in enumerate(freq_items)}
            self.categorical_mappings[col] = mapping
            self.field_dims.append(len(mapping) + 1)  # +1 for UNK

        # Save field_dims and stats
        np.save(self.processed_dir / "field_dims.npy", np.array(self.field_dims, dtype=np.int32))

        stats_to_save = {
            "numerical_means": {c: s.mean for c, s in self.numerical_stats.items()},
            "numerical_stds": {c: s.std for c, s in self.numerical_stats.items()},
            "categorical_mappings": self.categorical_mappings,
            "field_dims": self.field_dims,
        }
        with open(self.processed_dir / "feature_stats.pkl", "wb") as f:
            pickle.dump(stats_to_save, f)

        print(f"  Fit complete: {len(NUMERICAL_COLS)} numerical fields, "
              f"cat vocab sizes: {dict(zip(CATEGORICAL_COLS, self.field_dims))}")

    def transform(self):
        """Pass 2: re-stream, transform, and write to memmap files."""
        total_rows = self._count_rows()
        chunk_size = self.config.data.chunk_size
        n_chunks = (total_rows + chunk_size - 1) // chunk_size

        # Pre-allocate memmap files
        numerical_path = self.processed_dir / "numerical.npy"
        categorical_path = self.processed_dir / "categorical.npy"
        labels_path = self.processed_dir / "labels.npy"

        numer_mmap = np.memmap(numerical_path, dtype=np.float32, mode="w+",
                                shape=(total_rows, self.num_numerical))
        categ_mmap = np.memmap(categorical_path, dtype=np.int32, mode="w+",
                                shape=(total_rows, self.num_categorical))
        label_mmap = np.memmap(labels_path, dtype=np.float32, mode="w+",
                               shape=(total_rows,))

        print(f"Pass 2 (transform): writing memmap files ({n_chunks} chunks)...")

        row_offset = 0
        use_log = self.config.preprocessing.numerical.log_transform
        use_std = self.config.preprocessing.numerical.standardize

        for chunk in tqdm(
            pd.read_csv(
                self.raw_path,
                sep="\t",
                header=None,
                names=COLUMN_NAMES,
                chunksize=chunk_size,
                dtype={col: str for col in CATEGORICAL_COLS},
            ),
            total=n_chunks,
        ):
            n = len(chunk)
            end = row_offset + n

            # Numerical features
            numer_chunk = np.zeros((n, self.num_numerical), dtype=np.float32)
            for i, col in enumerate(NUMERICAL_COLS):
                vals = pd.to_numeric(chunk[col], errors="coerce").fillna(0).values.astype(np.float32)
                vals = np.clip(vals, 0, None)  # clip negative
                if use_log:
                    vals = np.log1p(vals)
                if use_std:
                    mean = self.numerical_stats[col].mean
                    std = max(self.numerical_stats[col].std, 1e-8)
                    vals = (vals - mean) / std
                numer_chunk[:, i] = vals
            numer_mmap[row_offset:end] = numer_chunk

            # Categorical features
            categ_chunk = np.zeros((n, self.num_categorical), dtype=np.int32)
            for i, col in enumerate(CATEGORICAL_COLS):
                mapping = self.categorical_mappings[col]
                categ_chunk[:, i] = chunk[col].fillna("").astype(str).map(
                    lambda x: mapping.get(x, 0)
                ).values.astype(np.int32)
            categ_mmap[row_offset:end] = categ_chunk

            # Labels
            label_mmap[row_offset:end] = chunk["label"].values.astype(np.float32)

            row_offset = end

        # Flush memmaps to disk
        numer_mmap.flush()
        categ_mmap.flush()
        label_mmap.flush()

        print(f"  Transform complete: wrote {total_rows:,} rows")

    def _create_splits(self):
        """Create shuffled train/val/test index splits."""
        total_rows = self._count_rows()
        rng = np.random.default_rng(self.config.data.random_seed)
        indices = rng.permutation(total_rows)

        train_end = int(total_rows * self.config.data.train_ratio)
        val_end = train_end + int(total_rows * self.config.data.val_ratio)

        np.save(self.processed_dir / "train_indices.npy", indices[:train_end].astype(np.int64))
        np.save(self.processed_dir / "val_indices.npy", indices[train_end:val_end].astype(np.int64))
        np.save(self.processed_dir / "test_indices.npy", indices[val_end:].astype(np.int64))

        print(f"Split: train={train_end:,} val={val_end - train_end:,} test={total_rows - val_end:,}")

    def run_pipeline(self):
        """Run the full preprocessing pipeline: fit → transform → splits."""
        print("=" * 60)
        print("Criteo Preprocessing Pipeline")
        print("=" * 60)

        # Check if already processed
        if (self.processed_dir / "numerical.npy").exists():
            print("Processed data already exists. Skipping preprocessing.")
            return

        print(f"Raw data: {self.raw_path}")
        print(f"Output dir: {self.processed_dir}")
        print()

        self.fit()
        print()
        self.transform()
        print()
        self._create_splits()
        print()
        print("Preprocessing complete.")
