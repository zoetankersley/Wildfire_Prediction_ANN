# Wildfire_Prediction_ANN

Yay!

## MLP regression and feature ablation

From the repository root, install the dependencies and run the experiment:

```sh
python -m pip install -r requirements.txt
python -m src.train_mlp --data data/clean_ca_2016_2020_mlp.parquet --output docs/mlp_results.json
```

Useful flags: `--seeds 42 43 44` (training seeds), `--random-trials 20`
(random-search configs), `--max-epochs`, `--patience`. For a quick check use
`--random-trials 0 --seeds 42`.

The `_mlp.parquet` file is a copy produced with the current
`notebooks/data_cleaning.ipynb`, including its eight NDVI summary columns. To
regenerate the inputs, place all five raw CSVs under `data/` with names ending
in `_FPA_FOD_cons.csv` and run the cleaning notebook. Then omit `--data` to use
its default Parquet output. The runner records the input SHA-256 and the actual
feature lists so runs on different data can be told apart.

### Method

- **Target:** `log1p(FIRE_SIZE)`. Acre-scale errors use inverse-transformed
  predictions clipped at zero.
- **Split:** the regional split from `split(df, seed=42)`, the same one saved in
  `docs/region_split_seed42.json`. Whole regions are held out. The runner checks
  its regions against that file and fails if they differ. Median imputation and
  scaling use training rows only.
- **Network:** input, two ReLU hidden layers (64, 32), one output. AdamW, MSE
  loss (Huber also tried), early stopping on validation log RMSE with the best
  epoch's weights restored.
- **Ablation:** Model 1 weather, Model 2 weather + vegetation, Model 3 weather +
  vegetation + geography.
- **Tuning:** done on Model 3 with the first seed, using validation data only.
  Nine named one-at-a-time variations (loss, width, depth, learning rate,
  batch size, dropout, weight decay, batch norm) plus a seeded random search over
  width, depth (1-4), learning rate, batch size, loss, dropout, weight decay and
  batch norm. The config with the lowest validation log RMSE is frozen.
- **Final runs:** Models 1-3 each train with the frozen config over several
  seeds. Held-out metrics are reported as mean and standard deviation.
- **Reference:** the JSON also holds the metrics of predicting the training mean,
  which any model must beat.
- **Reproducibility:** `docs/mlp_results.json` stores seeds, every config, the
  data hash, git commit, package versions, feature lists, per-epoch histories
  and metrics. Runs use deterministic algorithms and one CPU thread.

### Results

Not yet recorded for the current runner. Run the command above and add the
held-out log RMSE (mean ± std over seeds) for each model, next to the
mean-predictor reference, from `docs/mlp_results.json`.

## Baseline models

Use the same Parquet path for the baselines and the MLP to compare scores on
identical rows and features.

```sh
python -m src.train_baselines --data data/clean_ca_2016_2020_mlp.parquet
```

The runner fits all baselines on the training split for each of the three MLP
feature groups and reports validation and held-out test scores.

- Regression uses the training mean, training median, linear regression, and
  random forest. Compare log RMSE with the MLP, using the same feature group.
- Classification uses the training majority class, logistic regression, and
  random forest. It reports accuracy, balanced accuracy, and macro F1.
  The default is the project's `TARGET` label, which merges D/E and F/G.
  Pass `--classification-target FIRE_SIZE_CLASS` for the seven NWCG classes.

Results print as JSON. Pass `--output` with a path to save them. The report
includes the input hash, regional split, and feature lists for comparison with
the MLP report.
