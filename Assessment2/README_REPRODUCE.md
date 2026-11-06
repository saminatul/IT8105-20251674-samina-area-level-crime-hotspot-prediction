# How Assessment 2 was produced — step-by-step guide

Everything in the report comes from the code in `Assessment2/code`, which reads the raw CSV in
`machinelearningcode/Coding/Dataset/`. Your original notebooks (`machinelearningcode/Coding/Notebooks`, 01–09) are
the Assessment 1 **pilot** and were **not modified**; they were only audited (step 1).

## 1. Tools

| Tool | Used for | Version used |
|---|---|---|
| Python | everything | 3.9.6 |
| numpy, pandas, scipy | data handling | 2.0.2 / 2.3.3 / 1.13.1 |
| pyproj | lat/lon → UTM zone 16N (metres) | 3.6.1 |
| scikit-learn | LR, SVM, RF, KDE, DBSCAN, metrics | 1.6.1 |
| imbalanced-learn | SMOTE (tabular models only) | 0.12.4 |
| PyTorch + PyTorch Geometric | GCN, ResGCN, GAT, MLP | 2.8.0 / 2.6.1 |
| matplotlib | figures | — |
| joblib | saving the prototype model | — |
| tectonic (or pdflatex + bibtex) | compiling the LaTeX report | — |
| curl | downloading the extra 2-year dataset | — |

```bash
python3 -m pip install --user numpy pandas scipy scikit-learn "imbalanced-learn==0.12.4" \
    torch torch-geometric pyproj matplotlib joblib pillow
brew install tectonic          # macOS; or install TeX Live and use pdflatex + bibtex
```

## 2. Files

```
Assessment2/
  code/prep.py             cleaning, projection, 2.2 km grid, graph, leakage-safe features
  code/models.py           metrics, KDE, LR/SVM/RF wrappers, GCN, ResGCN, GAT, MLP, training loop
  code/run_experiments.py  all experiments E0–E19 (+ E21 via env var)
  code/multiyear.py        loader for the 3-year data
  code/figures.py          all figures
  app/score_batch.py       batch scoring job (prototype)
  app/server.py            read-only API + dashboard (prototype)
  app/test_app.py          19 automated tests + latency
  results/ results_multi/  CSV/JSON output of every experiment
  figures/                 PNG figures
  latex/main.tex, references.bib, figures/   the report
```

## 3. What was done, in order (with commands)

Run every command from `Assessment2/code` unless stated.

**Step 1 – Audit the Assessment 1 pilot (E0).**
Checked the numbers in the proposal against the notebooks and CSVs: cleaning counts, the Phase 1 label (it is the
arg-max of the input features), the string-valued arrest/domestic columns, constant edge weights, and re-created the
pilot's Phase 2 split to show LR/persistence score 100%.
```bash
python3 prep.py                          # prints raw -> cleaned counts (229,275 -> 228,595), 157 cells, 272 edges
python3 run_experiments.py E0            # writes results/E0_pilot_audit.json
```

**Step 2 – Clean the data.** Drop 2 exact duplicates, 40 rows without coordinates, 638 rows outside
lat 41.6–42.1 / lon −87.9…−87.5 → 228,595 rows (all in `prep.clean`).

**Step 3 – Grid and graph.** Project to EPSG:26916, 2,200 m square cells (origin = min x/y), keep occupied cells (157),
connect cell centres within 3 km (KD-tree) → 272 undirected edges; every edge is 2.2 km (rook lattice).

**Step 4 – Periods, features, targets (no leakage).** 28-day periods (13 in the main data). For target period *t*,
features use periods < *t* only (23 features: lags, long-run means, shares, trend, coordinates). Targets:
Phase 2 = cell is in top 20% of victimisation counts in period *t*; Phase 1 = dominant crime type (4 classes, rule
derived from training periods only).

**Step 5 – Chronological split.** Train target periods 3–8, validation 9–10, test 11–12 (test never used for choices).

**Step 6 – Tune and test all models (E1, E3).** Persistence, KDE, LR, SVM, RF, MLP, GCN, ResGCN, GAT; validation-only
grid search, 10 seeds on test.
```bash
python3 run_experiments.py E1 E3         # ~4 min; results/E2_*.csv, E3_*.csv, best-param JSONs
```

**Step 7 – Robustness and ablations (E4–E14).** Imbalance (weights/SMOTE/threshold), graph radius/weights/depth,
feature groups, label percentile and grid size, rolling origin, spatial-block CV, block bootstrap, count regression,
DBSCAN, runtime, feature importance.
```bash
python3 run_experiments.py E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 curves E14   # ~10 min
```

**Step 8 – Post-hoc extensions (E15–E19).** Preprocessing ablation, permutation tests, extra Phase 1 features,
tuned count models, effective sample size + centroid graph. (Labelled post-hoc in the report.)
```bash
python3 run_experiments.py E15 E16 E17 E18 E19                            # ~7 min
```
To run everything for the main data in one go: `python3 run_experiments.py` (≈20 min; results are deterministic).

**Step 9 – Three-year replication (E21).** Download the extra two years (24 Aug 2023 – 23 Aug 2025) from the official
City of Chicago API (≈510,000 rows, ≈106 MB), then rerun with `A2_DATA=multi`:
```bash
cd ../../machinelearningcode/Coding/Dataset
curl -sG "https://data.cityofchicago.org/resource/ijzp-q8t2.csv" \
  --data-urlencode '$select=id,case_number,date,block,iucr,primary_type,description,location_description,arrest,domestic,beat,ward,fbi_code,x_coordinate,y_coordinate,latitude,longitude' \
  --data-urlencode "\$where=date >= '2023-08-24T00:00:00' AND date < '2025-08-24T00:00:00'" \
  --data-urlencode '$order=id' --data-urlencode '$limit=600000' \
  -o Crimes_2001_to_present_subset_20230824_20250823.csv
cd ../../../Assessment2/code && mkdir -p ../results_multi
A2_DATA=multi python3 -u run_experiments.py E1 E3 E4 E8 E9 E10 E16 E19 E13      # ~25 min
A2_DATA=multi A2_ROLL=15:39:2 A2_SUFFIX=_long python3 -u run_experiments.py E8  # ~15 min, 12 rolling windows
python3 multiyear.py                     # prints per-period counts to check the join between the two files
```

**Step 9b – Extension experiments (E22–E24).** Violent-only label (FBI codes 01A, 02, 03, 04A, 04B), error-disparity
audit by community-area hardship, and gradient boosting. E23 needs two public City of Chicago files in
`machinelearningcode/Coding/Dataset/external/` (census indicators kn9c-c2s2 as CSV; community-area boundaries igwz-8jzy as GeoJSON).
```bash
curl -o ../../machinelearningcode/Coding/Dataset/external/census_socioeconomic_2008_2012.csv \
  "https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD"
curl -o ../../machinelearningcode/Coding/Dataset/external/community_areas.geojson \
  "https://data.cityofchicago.org/api/geospatial/igwz-8jzy?method=export&format=GeoJSON"
python3 extensions.py                    # ~2 min; results/E22_*, E23_*, E24_*
```

**Step 10 – Figures.**
```bash
python3 figures.py                       # writes ../figures/F01…F15 (run after the experiments)
```

**Step 11 – Prototype application (E20).** Batch job → API → tests.
```bash
cd ../app
python3 score_batch.py                   # fits LR as in E2, scores all cells/periods, writes artifacts/
python3 test_app.py                      # 19 tests, parity with offline results, latency -> results/E20_prototype_test.json
A2_KEYS="mykey:analyst" python3 server.py --port 8080      # run the API yourself
curl -H "X-API-Key: mykey" "http://127.0.0.1:8080/v1/forecasts?period=13&top_k=32"
curl -H "X-API-Key: mykey" "http://127.0.0.1:8080/dashboard?period=13" > dashboard.html && open dashboard.html
```
(Browsers cannot send the `X-API-Key` header by themselves, so save the dashboard with curl as shown.)

**Step 12 – Write and compile the report.**
```bash
cd ../latex
tectonic main.tex                        # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 4. Experiment map

| ID | Question | Main output |
|---|---|---|
| E0 | Is the pilot valid? | `E0_pilot_audit.json` |
| E1/E2 | Phase 2 tuning and test | `E1_p2_*`, `E2_p2_test_*` |
| E3 | Phase 1 tuning and test | `E3_p1_*` |
| E4 | Class weights / SMOTE / thresholds | `E4_imbalance.csv` |
| E5 | Graph radius, weights, depth | `E5_graph_ablation.csv` |
| E6 | Feature-group ablation | `E6_feature_ablation.csv` |
| E7 | Label percentile, grid size | `E7_sensitivity.csv` |
| E8 | Rolling origin | `E8_rolling_origin*.csv` |
| E9 | Spatial-block CV | `E9_spatial_block_cv.csv` |
| E10 | Block bootstrap CIs | `E10_bootstrap.csv` |
| E11/E18 | Count regression (untuned / tuned) | `E11_*`, `E18_*` |
| E12 | DBSCAN hotspots, PAI | `E12_*` |
| E13 | Runtime, memory | `E13_environment.json` |
| E14 | Feature importance | `E14_feature_importance.csv` |
| E15 | Preprocessing ablation | `E15_preprocessing_ablation.csv` |
| E16 | Permutation tests | `E16_permutation_tests.csv` |
| E17 | Extra Phase 1 features | `E17_phase1_features.csv` |
| E19 | Effective sample size, centroid graph | `E19_*.json` |
| E20 | Prototype tests | `E20_prototype_test.json` |
| E21 | Three-year replication | `results_multi/` |

## 5. Tips and pitfalls

- Seeds are fixed, so reruns give identical numbers (verified for E1, E3, E4, E10, E11, E12).
- Harmless `RuntimeWarning: ... encountered in matmul` messages come from numpy on Apple silicon; ignore them.
- Run `E1` and `E3` before experiments that read their outputs (E10, E12, E15–E19).
- For background runs that write a log into `results_multi/`, create the folder first (`mkdir -p ../results_multi`).
- The scripts read the CSV from `machinelearningcode/Coding/Dataset/`; keep that folder layout.
- Do not commit API keys: the tests generate throw-away keys at run time.
