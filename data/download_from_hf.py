"""Download the Criteo dataset from HuggingFace (via hf-mirror.com) and
save as train.txt in the expected TSV format for the preprocessing pipeline.

Expected output format (tab-separated):
    label  I1  I2  ...  I13  C1  C2  ...  C26

Integer features: use 0 for missing values (None).
Categorical features: hex strings or "" for None.
"""

import os
import sys
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

OUTPUT_PATH = "data/train.txt"
EXPECTED_ROWS = 45_840_617  # Criteo Display Advertising Challenge train set


def format_row(row):
    """Convert a HuggingFace dataset row to TSV format."""
    label = str(row["label"])

    # Integer features (I1-I13): replace None with 0
    int_features = []
    for i in range(1, 14):
        val = row[f"integer_feature_{i}"]
        int_features.append(str(val) if val is not None else "0")

    # Categorical features (C1-C26): replace None with empty string
    cat_features = []
    for i in range(1, 27):
        val = row[f"categorical_feature_{i}"]
        cat_features.append(val if val is not None else "")

    return "\t".join([label] + int_features + cat_features)


def main():
    print(f"Downloading Criteo dataset from {os.environ['HF_ENDPOINT']}...")
    print("Streaming the full 45M-row dataset and writing to TSV...")
    print(f"Output: {OUTPUT_PATH}\n")

    # Load in streaming mode to avoid downloading everything into memory
    dataset = load_dataset(
        "criteo/CriteoClickLogs",
        split="train",
        streaming=True,
    )

    row_count = 0
    total_bytes = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in tqdm(dataset, total=EXPECTED_ROWS, desc="Downloading & converting"):
            line = format_row(row) + "\n"
            f.write(line)
            total_bytes += len(line)
            row_count += 1

            # Progress checkpoint every 5M rows
            if row_count % 5_000_000 == 0:
                print(f"  {row_count:,} rows written, ~{total_bytes / (1024**3):.1f} GB")

    print(f"\nDownload complete: {row_count:,} rows, {total_bytes / (1024**3):.2f} GB")
    print(f"Output file: {OUTPUT_PATH}")

    # Verify first few lines
    print("\nVerification — first 3 lines:")
    with open(OUTPUT_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            cols = line.strip().split("\t")
            print(f"  {len(cols)} columns: {cols[:5]}... (label={cols[0]})")

    print(f"\nExpected: {EXPECTED_ROWS:,} rows, 39 columns")
    if row_count == EXPECTED_ROWS:
        print("Row count matches expected!")
    else:
        print(f"Warning: got {row_count:,} rows, expected {EXPECTED_ROWS:,}")


if __name__ == "__main__":
    main()
