"""Run the Models 1–3 feature ablation on the team's shared regional split.

Workflow (nothing is trained until you run this module):
  1. Tune on Model 3 (richest features) with one seed: named one-at-a-time
     variations of the baseline plus a seeded random search.
  2. Freeze the config with the lowest validation log RMSE.
  3. Train Models 1-3 with that config over several seeds and report mean/std.
Test data is only touched in step 3.
"""

import argparse
from dataclasses import asdict, replace
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pandas as pd
import torch

from .data import (
    DATA_PATH, GROUP, SPLIT_PATH, feature_sets, log_target, prepare_features, split,
)
from .evaluate import regression_metrics
from .mlp import MLPConfig, fit_mlp, predict_log

SPLIT_SEED = 42  # matches docs/region_split_seed42.json made in notebooks/data_splitting.ipynb
MODEL_NAMES = ("model1_weather", "model2_weather_veg", "model3_all")
METRICS = ("rmse_log", "mae_log", "r2_log", "rmse_acres", "mae_acres")


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_saved_split(parts):
    """Fail if our regional split differs from the team's saved assignment."""
    if not SPLIT_PATH.exists():
        return "saved split file not found; check skipped"
    saved = json.loads(SPLIT_PATH.read_text())
    for label, part in zip(("train", "val", "test"), parts):
        wrong = [r for r in part[GROUP].unique() if saved.get(r) != label]
        if wrong:
            raise ValueError(
                f"{label} regions disagree with {SPLIT_PATH.name}: {wrong[:5]}..."
            )
    return "matches docs/region_split_seed42.json"


def candidate_configs(baseline, n_random, search_seed):
    """Named one-at-a-time variations, then a seeded random search."""
    named = {
        "baseline_mse": baseline,
        "huber": replace(baseline, loss="huber"),
        "wider": replace(baseline, hidden_sizes=(128, 64)),
        "deeper": replace(baseline, hidden_sizes=(64, 32, 16)),
        "lower_learning_rate": replace(baseline, learning_rate=0.0003),
        "smaller_batch": replace(baseline, batch_size=128),
        "dropout": replace(baseline, dropout=0.15),
        "weight_decay": replace(baseline, weight_decay=0.001),
        "batch_norm": replace(baseline, batch_norm=True),
    }
    rng = random.Random(search_seed)
    for i in range(n_random):
        width, depth = rng.choice((32, 64, 128, 256)), rng.choice((1, 2, 3, 4))
        named[f"random_{i:02d}"] = replace(
            baseline,
            hidden_sizes=tuple(max(width // 2**d, 8) for d in range(depth)),
            learning_rate=10 ** rng.uniform(-4, -2.5),
            batch_size=rng.choice((64, 128, 256, 512)),
            loss=rng.choice(("mse", "huber")),
            dropout=rng.choice((0.0, 0.1, 0.2, 0.3, 0.5)),
            weight_decay=rng.choice((0.0, 1e-5, 1e-4, 1e-3, 1e-2)),
            batch_norm=rng.random() < 0.5,
        )
    return named


def summarize(runs):
    """Mean and sample std of each held-out metric over seeds."""
    return {
        m: {
            "mean": float(np.mean([r["test"][m] for r in runs])),
            "std": float(np.std([r["test"][m] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
        }
        for m in METRICS
    }


def run_ablation(data_path, seeds, baseline, n_random):
    with data_path.open("rb") as data_file:
        data_sha256 = file_digest(data_file, "sha256").hexdigest()
    df = pd.read_parquet(data_path)
    valid = np.isfinite(df["FIRE_SIZE"]) & (df["FIRE_SIZE"] >= 0)
    parts = split(df.loc[valid], seed=SPLIT_SEED)
    split_check = check_saved_split(parts)
    y_train, y_val, y_test = (log_target(part).to_numpy() for part in parts)
    groups = feature_sets(df)
    inputs = {}
    for name in MODEL_NAMES:
        # Median imputation cannot learn a value for columns empty in training.
        groups[name] = [col for col in groups[name] if parts[0][col].notna().any()]
        inputs[name] = prepare_features(*parts, groups[name])

    # Model 0: predict the training mean. Anything the MLP scores must beat this.
    mean_pred = np.full(len(y_test), y_train.mean())
    reference = {
        "validation": regression_metrics(y_val, np.full(len(y_val), y_train.mean())),
        "test": regression_metrics(y_test, mean_pred),
    }
    print(f"mean-predictor {reference['validation']['rmse_log']:.4f} validation log RMSE")

    def train_trial(name, config, seed):
        x_train, x_val, x_test = inputs[name]
        model, best_epoch, history = fit_mlp(x_train, y_train, x_val, y_val, config, seed)
        return {
            "seed": seed,
            "model": name,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "validation": regression_metrics(y_val, predict_log(model, x_val)),
            "test": regression_metrics(y_test, predict_log(model, x_test)),
            "history": history,
        }

    # Step 1: tune on Model 3 using validation only; test metrics are ignored here.
    tuning_model, tuning_seed = MODEL_NAMES[-1], seeds[0]
    trials, selected, best_score = {}, None, float("inf")
    for label, config in candidate_configs(baseline, n_random, tuning_seed).items():
        trial = train_trial(tuning_model, config, tuning_seed)
        trial.pop("test")  # keep test data out of selection and out of the log
        trials[label] = trial
        score = trial["validation"]["rmse_log"]
        print(f"tuning {label:22} {score:.4f} validation log RMSE", flush=True)
        if score < best_score:
            selected, best_score = label, score
    chosen = MLPConfig(**{**trials[selected]["config"],
                          "hidden_sizes": tuple(trials[selected]["config"]["hidden_sizes"])})

    # Step 2/3: freeze the config, train every feature group on every seed.
    ablation = {}
    for name in MODEL_NAMES:
        runs = [train_trial(name, chosen, seed) for seed in seeds]
        ablation[name] = {"runs": runs, "test_summary": summarize(runs)}
        s = ablation[name]["test_summary"]["rmse_log"]
        print(f"{name} {s['mean']:.4f} ± {s['std']:.4f} held-out log RMSE", flush=True)

    return {
        "data_sha256": data_sha256,
        "git_commit": git_commit(),
        "split_seed": SPLIT_SEED,
        "split_check": split_check,
        "training_seeds": seeds,
        "target": "log1p(FIRE_SIZE)",
        "split": {
            label: {"rows": len(part), "regions": sorted(part[GROUP].unique().tolist())}
            for label, part in zip(("train", "validation", "test"), parts)
        },
        "excluded_invalid_targets": int((~valid).sum()),
        "features": {name: groups[name] for name in MODEL_NAMES},
        "versions": {
            package: version(package)
            for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch")
        },
        "mean_predictor_reference": reference,
        "selected_config": selected,
        "tuning_model": tuning_model,
        "tuning_seed": tuning_seed,
        "tuning": trials,
        "ablation": ablation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("mlp_results.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--random-trials", type=int, default=20,
                        help="random-search configs on top of the 9 named ones")
    parser.add_argument("--max-epochs", type=int, default=MLPConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=MLPConfig.patience)
    args = parser.parse_args()
    if min(args.max_epochs, args.patience) < 1 or args.random_trials < 0:
        parser.error("--max-epochs and --patience must be positive; --random-trials >= 0")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = MLPConfig(max_epochs=args.max_epochs, patience=args.patience)
    results = run_ablation(args.data, args.seeds, baseline, args.random_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
