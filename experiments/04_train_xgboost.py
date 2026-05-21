"""Step 4: Train XGBoost baseline."""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepfm_ctr.config import load_config
from deepfm_ctr.training.evaluator import evaluate_all
from deepfm_ctr.utils.reproducibility import seed_everything
from deepfm_ctr.visualization.plotter import plot_prediction_distribution


def load_split_data(processed_dir, split):
    """Load a split into memory using memmap indexing (efficient)."""
    processed_dir = Path(processed_dir)
    indices = np.load(processed_dir / f"{split}_indices.npy").astype(np.int64)

    n_rows = int((processed_dir / "numerical.npy").stat().st_size / (13 * 4))
    shape_num = (n_rows, 13)
    shape_cat = (shape_num[0], 26)

    numer_mmap = np.memmap(processed_dir / "numerical.npy", dtype=np.float32, mode="r", shape=shape_num)
    categ_mmap = np.memmap(processed_dir / "categorical.npy", dtype=np.int32, mode="r", shape=shape_cat)
    label_mmap = np.memmap(processed_dir / "labels.npy", dtype=np.float32, mode="r")

    sorted_indices = np.sort(indices)
    return (
        np.array(numer_mmap[sorted_indices], dtype=np.float32),
        np.array(categ_mmap[sorted_indices], dtype=np.int32),
        np.array(label_mmap[sorted_indices], dtype=np.float32),
    )


def main():
    import xgboost as xgb

    config = load_config("configs/default.yaml")
    seed_everything(config.data.random_seed)

    processed_dir = config.data.processed_dir
    xgb_config = config.baselines.xgboost

    print("Loading preprocessed data for XGBoost...")
    X_num_train, X_cat_train, y_train = load_split_data(processed_dir, "train")
    X_num_val, X_cat_val, y_val = load_split_data(processed_dir, "val")
    X_num_test, X_cat_test, y_test = load_split_data(processed_dir, "test")

    # Concatenate numerical and categorical features
    X_train = np.concatenate([X_num_train, X_cat_train.astype(np.float32)], axis=1)
    X_val = np.concatenate([X_num_val, X_cat_val.astype(np.float32)], axis=1)
    X_test = np.concatenate([X_num_test, X_cat_test.astype(np.float32)], axis=1)

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": xgb_config.max_depth,
        "learning_rate": xgb_config.learning_rate,
        "subsample": xgb_config.subsample,
        "colsample_bytree": xgb_config.colsample_bytree,
        "seed": config.data.random_seed,
        "verbosity": 1,
    }

    print("Training XGBoost...")
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params, dtrain, num_boost_round=xgb_config.n_estimators,
        evals=evals, early_stopping_rounds=xgb_config.early_stopping_rounds,
        verbose_eval=20,
    )

    # Evaluate
    scores = model.predict(dtest)
    metrics = evaluate_all(y_test, scores)
    print(f"\nTest results: AUC={metrics['auc']:.4f} LogLoss={metrics['logloss']:.4f}")

    with open("results/xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    import pickle as pkl
    model.save_model("results/checkpoints/xgboost_best.model")
    with open("results/checkpoints/xgboost_best.pkl", "wb") as f:
        pkl.dump({"model": model, "test_labels": y_test}, f)

    plot_prediction_distribution(y_test, scores, "XGBoost", "results/figures/xgboost_pred_dist.png")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    print(f"\nTop 10 important features: {sorted(importance.items(), key=lambda x: -x[1])[:10]}")


if __name__ == "__main__":
    main()
