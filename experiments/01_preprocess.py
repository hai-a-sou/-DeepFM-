"""Step 1: Run the full Criteo preprocessing pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepfm_ctr.config import load_config
from deepfm_ctr.data.preprocessing import CriteoPreprocessor


def main():
    config = load_config("configs/default.yaml")
    preprocessor = CriteoPreprocessor(config)
    preprocessor.run_pipeline()


if __name__ == "__main__":
    main()
