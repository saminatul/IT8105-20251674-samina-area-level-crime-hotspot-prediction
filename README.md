# Area-level crime hotspot and high-victimisation risk prediction

**IT8105 Machine Learning, Assessment 2.** Scenario-based evaluative report of a machine-learning solution for short-horizon, area-level forecasting of recorded crime in the City of Chicago.

| | |
|---|---|
| Student | Samina Tuladhar |
| Student ID | 20251674 |
| Institution | Whitecliffe |
| Lecturer | Dr Joseph Dang |
| Submission | 6 November 2026 |
| Report | [`Assessment2/latex/20251674_IT8105_Assess2.pdf`](Assessment2/latex/20251674_IT8105_Assess2.pdf) |
| Report source | [`Assessment2/latex/main.tex`](Assessment2/latex/main.tex), [`Assessment2/latex/references.bib`](Assessment2/latex/references.bib) |
| Reproduction | [`README_REPRODUCE.md`](README_REPRODUCE.md) |

The study asks whether a spatial graph convolutional network forecasts the next planning period more accurately than persistence, kernel density, and standard classifiers, when every model sees the same leakage-safe features, labels, chronological split, and tuning budget.

Outputs are **area-level counts of recorded incidents**. They are decision support for crime and community-safety analysts. They are counts of recorded events, they are not unique victims, they are not adjusted for population or footfall, and they are not an individual’s probability of harm. A human reviews every use. Nothing in this repository assigns patrol or outreach automatically.

---

## Forecasting tasks

The prediction unit is a **2.2 km × 2.2 km** grid cell. Each forecast covers the **next 28 days**.

For cell \(i\) in period \(t\), features \(\mathbf{x}_{i,t}\) are built only from periods before \(t\). Graph models may also use the features of spatial neighbours. The two tasks are:

1. **Phase 1 — dominant crime type.** The most frequent primary category in the cell during period \(t\). Training periods keep three classes that dominate at least 5% of training cell-periods (**theft**, **battery**, **criminal damage**). Every other category is **other type**. This describes the mix of recorded offending. It does not measure hotspot intensity.
2. **Phase 2 — high recorded victimisation.** Whether the cell is among the top 20% of the 157 cells that period for person-directed incidents: assault, battery, robbery, criminal sexual assault, sex offence, stalking, homicide, kidnapping, offence involving children, intimidation, and human trafficking (73,494 records, 32.2% of the cleaned extract). Ties at the cutoff give a prevalence of about 20.4–22.3% (32–35 cells). At prediction time the program ranks cells and flags the top 32. No future outcome is used to set that cutoff.

Consecutive periods have a Spearman correlation of about **0.965–0.977** in cell victimisation counts, so a recency-weighted persistence rule is the standard any learned model has to beat. In this extract the busiest 10% of occupied cells hold about 30% of incidents, and the busiest 20% hold about 49%.

---

## Research questions and objectives

The question carried forward from Assessment 1 is whether a spatial graph convolutional network improves area-level forecasts of crime type and recorded victimisation beyond conventional spatial and machine-learning methods.

| ID | Question | Answer on the held-out periods |
|---|---|---|
| RQ1 | Does a GCN forecast next-period dominant crime type better than a non-graph SVM and matched baselines? | No. Best macro-F1 is logistic regression at **0.498**. The GCN is **0.450**. |
| RQ2 | Can history-only features identify top-20% victimisation cells with acceptable precision and recall, and does the graph add value? | Yes for history-only features. Logistic regression reaches precision **0.938** and recall **0.909**. The graph does not add value. |
| RQ3 | How much of the measured skill is real signal rather than chance, given few cells and spatial dependence? | The skill is real and is mostly persistence. The gain of learned models over persistence (about 0.01 average precision) sits inside the uncertainty of roughly 47–50 effective test observations. |

| ID | Objective | Status |
|---|---|---|
| RO1 | Reproducible pipeline: reconciled counts, documented grid and graph, features from earlier periods only | Achieved |
| RO2 | Forecast next-period dominant crime type; compare a GCN with an SVM and matched baselines on macro-F1 in periods 11–12 | Partly achieved. Forecasting is evaluated. No model does well, and the GCN does not beat the SVM |
| RO3 | Identify top-20% recorded-victimisation cells with macro-F1 above 0.70 on held-out periods | Achieved (macro-F1 **0.951**). The 0.70 bar is too weak to separate models, because persistence itself scores **0.932** |
| RO4 | Robustness: 10 seeds, spatial-block bootstrap, rolling origin, spatial-block testing, ablations, sensitivity | Achieved, including a three-year replication |
| RO5 | A usable, governed application, with practical value and ethical risk examined | Design completed and prototype tested. Not deployed |

---

## From the Assessment 1 pilot to this study

[`machinelearningcode/`](machinelearningcode/) holds the Assessment 1 pilot: notebooks 01–09, their saved tables, and the background papers. Those notebooks were **audited and left unchanged**. The final numbers in the report come from [`Assessment2/code/`](Assessment2/code/), which rebuilds the panel from the raw CSV.

The pilot audit (experiment E0) changed the scope of the final study:

| Pilot claim | What the audit found |
|---|---|
| Cleaning 229,275 → 228,595, explained as 2 duplicates and 638 boundary drops | The arithmetic also needs **40 rows with missing coordinates**. The stage log now matches exactly |
| Phase 1 GCN accuracy 66.67% (macro-F1 0.218) against an SVM at 37.50% | The Phase 1 label is the arg-max of the same-period crime-type counts used as features. The rule recovers the label of all 157 cells. The task was circular |
| Phase 2 GCN on a random 24-cell test (5 positive cells): accuracy 0.875, macro-F1 0.842 | Class-weighted logistic regression, and a persistence rule, both score **1.000** on that same split. Past and future victimisation rates correlate at Spearman **0.989**. One false negative moves recall by 20 points |
| Inverse-distance edge weights on a 3 km graph | All 272 undirected edges are 2.2 km, so the weights are constant. The pilot GCN did not pass edge weights into the convolution |
| Arrest and domestic history | The saved pilot file stored concatenated `Y`/`N` strings. The final pipeline converts flags to 0/1 before summing |

The final study therefore treats both phases as one-period-ahead forecasts, with history-only features, matched baselines, and a chronological test.

---

## Data

**Main extract.** Chicago Police Department *Crimes — One year prior to present*, City of Chicago Data Portal, downloaded 31 August 2026.

- File: [`machinelearningcode/Coding/Dataset/Crimes_-_One_year_prior_to_present_20260831.csv`](machinelearningcode/Coding/Dataset/Crimes_-_One_year_prior_to_present_20260831.csv)
- 17 fields, **229,275** rows, 24 August 2025 05:03 to 23 August 2026 00:00
- The portal describes the extract as preliminary. It can be reclassified, it omits the most recent seven days, and locations are released at block level
- Used for non-commercial research. The prototype dashboard states that the data have been modified from the City of Chicago source

**Cleaning** (all in [`Assessment2/code/prep.py`](Assessment2/code/prep.py)):

| Stage | Rule | Remaining | Removed |
|---|---|---:|---:|
| Raw | All timestamps parsed | 229,275 | — |
| Exact duplicates | Identical on all 17 fields | 229,273 | 2 |
| Missing coordinates | Latitude and longitude missing (the same 40 rows lack X/Y) | 229,233 | 40 |
| Study boundary | Latitude 41.6–42.1, longitude −87.9 to −87.5 | **228,595** | 638 |

`LOCATION DESCRIPTION` is missing on 1,037 rows (0.45%) and is set to `UNKNOWN`. One missing ward is unused. Twenty-two repeated case numbers are kept, because one case can carry several offence records. High-count cells are kept: concentration is the phenomenon being modelled.

**Fields that enter the model.** Date of occurrence, primary description, arrest, domestic, latitude, and longitude. Block, IUCR, FBI code, beat, and ward are not model inputs in the main pipeline. FBI codes are used only in the violent-crime label check (E22).

**Three-year replication (E21).** The same portal’s *Crimes — 2001 to Present* supplies 509,905 records from 24 August 2023 to 23 August 2025 (`Crimes_2001_to_present_subset_20230824_20250823.csv`, downloaded by the steps in [`README_REPRODUCE.md`](README_REPRODUCE.md)). Joined to the one-year extract this is **732,017** rows, 158 cells, and 274 undirected edges. The second CSV is not stored in this checkout until that download is run.

**External files for the hardship audit (E23).**

- [`machinelearningcode/Coding/Dataset/external/census_socioeconomic_2008_2012.csv`](machinelearningcode/Coding/Dataset/external/census_socioeconomic_2008_2012.csv) (community-area census indicators, 2008–2012)
- [`machinelearningcode/Coding/Dataset/external/community_areas.geojson`](machinelearningcode/Coding/Dataset/external/community_areas.geojson)

---

## Method

Thirteen periods of 28 days cover four full weeks, remove day-of-week composition effects, and match a monthly planning cycle. Periods 0–2 supply lags only. **Train** is target periods 3–8 (942 cell-periods), **validation** is 9–10 (314), and **test** is 11–12 (314 Phase 2 rows; 303 Phase 1 rows, because a cell-period with no incidents has no dominant type). The test window is 28 June to 22 August 2026 and is used once, after every modelling choice. The last calendar day of the extract (23 August 2026, 22 incidents) is counted in the cleaning log and is not a modelled period.

| Period | First day | Last day | Incidents | Role |
|---:|---|---|---:|---|
| 0 | 2025-08-24 | 2025-09-20 | 18,683 | features only |
| 1 | 2025-09-21 | 2025-10-18 | 18,960 | features only |
| 2 | 2025-10-19 | 2025-11-15 | 18,236 | features only |
| 3–8 | 2025-11-16 | 2026-05-02 | 16k–18k each | train |
| 9–10 | 2026-05-03 | 2026-06-27 | 18,970 and 18,399 | validation |
| 11–12 | 2026-06-28 | 2026-08-22 | 18,938 and 17,223 | test |

The three-year run uses the same 28-day clock from 24 August 2023: features-only periods 0–2, training targets 3–26, validation 27–32, test 33–38 (5 March to 19 August 2026).

**Grid.** WGS84 coordinates are projected to UTM zone 16N (EPSG:26916). The origin is the minimum easting and northing of the cleaned points (425,477 m E, 4,610,479 m N). Cell size is 2,200 m, which recovers the pilot’s **157 occupied cells** (1 to 9,549 incidents; median 1,080). Seventeen low-volume cells, mostly near the boundary or the lakeshore, stay in the panel so the ranking is not pulled toward the centre.

**Graph.** Cell centres within 3 km are joined with a KD-tree. That yields **272 undirected edges** (544 directed), no stored self-loops, and no isolated nodes. On this square lattice every edge is 2.2 km. Diagonal neighbours sit at 3.11 km and fall outside the radius, so the graph is a rook lattice (degree 1–4, mean 3.46). Each modelling period is a snapshot of the same graph. Snapshots are stacked as a block-diagonal graph so messages do not cross periods. At scoring time, features and edges are known and labels from the target period are not. That protocol is inductive in time.

**Features (23, history only).** For target period \(t\):

- Victimisation and total counts at lags 1, 2, and 3
- Long-run means of victimisation and of all incidents over every earlier period
- Shares: victimisation, domestic, arrest, and night-time (20:00–05:59)
- Crime mix over the last three periods (eight common types)
- Trend: lag 1 minus lag 3
- Cell-centre latitude and longitude

Counts use \(\log(1+x)\). The trend uses a signed logarithm. Every feature is then standardised with the mean and standard deviation of the **training rows only**. Two neighbour-average features exist only for the spatial-lag ablation. The full dictionary is Appendix A of the report and the feature list in `prep.py`.

**Why these checks, and what was skipped.** Class imbalance on Phase 2 is 1:4 by construction. Experiment E4 compares no correction, inverse-frequency weights, and SMOTE on logistic regression only. Synthetic nodes would need invented edges, so graph models use class-weighted cross-entropy instead. The operational rule (flag 32 cells) matches the prevalence and does not depend on a probability threshold. No automatic feature selection and no principal components: 23 interpretable features with L2 regularisation are enough, and components would remove the explanation the application needs. Importance (E14) is logistic-regression coefficients and permutation importance on the validation periods. The largest signal is the long-run victimisation mean.

**Phase 2 metrics.** Average precision (step-wise area under the precision–recall curve) is primary. The operational score is positive-class F1 and macro-F1 under the top-20% rule. ROC-AUC, a validation-tuned threshold, and the Brier score are secondary. A random ranking has average precision equal to the prevalence (0.21). An all-low classifier has macro-F1 0.44.

**Phase 1 metrics.** Macro-F1 over the four classes is primary, with accuracy, weighted F1, balanced accuracy, Cohen’s κ, and per-class F1. The majority-class (battery) macro-F1 is 0.134.

**Uncertainty.** Ten training seeds. A spatial-block bootstrap resamples 26 blocks of 3×3 cells, 2,000 times, keeping both test periods of a block together. Rolling-origin and five-block spatial cross-validation ask separate generalisation questions. Paired permutation tests (E16) are Holm-corrected. Random cell splits and stratified k-fold were not used: they mix future periods into training, and the rare Phase 1 classes cannot appear in every fold.

---

## Models

Every model in a phase shares labels, features, the chronological split, and a validation-only tuning budget. The GCN and the MLP share depth, width options, and the training loop, so a gap between them is the effect of message passing. Neural nets train with full-batch Adam for at most 400 epochs and early stopping (patience 60) on validation average precision (Phase 2) or macro-F1 (Phase 1).

| Model | Role | Validation choice |
|---|---|---|
| Majority | Null reference | — |
| Persistence | Recency-weighted victimisation lags \(1\cdot L_1 + 0.5\cdot L_2 + 0.25\cdot L_3\); Phase 1 uses the dominant category of the last three periods | — |
| KDE | Gaussian kernel density of the previous three periods, scored at cell centres. Phase 1 uses one count-scaled density per category | Bandwidth 800 m |
| Logistic regression | L2-regularised, balanced class weights. Selected Phase 2 model | \(C = 10\) (Phase 2), \(C = 100\) (Phase 1) |
| RBF-SVM | Proposal baseline | \(C = 10\), \(\gamma = 0.01\) (Phase 2) |
| Random forest | 300 trees, balanced-subsample weights | Unlimited depth, leaf 1 (Phase 2); leaf 5 (Phase 1) |
| MLP | Two linear layers, ReLU, dropout. Same code path as the GCN with no edges | Hidden 128, dropout 0.3 (Phase 2) |
| GCN | Two `GCNConv` layers, self-loops, symmetric normalisation. Proposal primary model | Hidden 64, dropout 0.3 (Phase 2) |
| GAT | Two `GATConv` layers, 4 heads. Learned neighbour weights | Hidden 64 (Phase 2), 128 (Phase 1) |
| ResGCN | GCN layer plus a separate linear path on the node’s own features, so the model can keep cell-specific signal | Hidden 128, dropout 0.3 (Phase 2) |
| DBSCAN | Unsupervised clusters of victimisation coordinates. Descriptive micro-hotspot layer | \(\varepsilon = 300\) m, minPts 60 |
| Poisson GLM and a Poisson GCN | Next-period counts (E11, retuned in E18) | — |
| Histogram gradient boosting | Post-hoc capacity check (E24) | Not selected: validation average precision 0.962 versus 0.968 for logistic regression |

On this rook lattice, symmetric normalisation gives a typical cell only about one fifth of the weight on its own history after one layer, and less after two. That is Laplacian smoothing. It helps when neighbours share a label and hurts when the label is the cell’s own rank. The residual self-path was added from that analysis, before the test periods were scored. Wider graphs and extra layers were ablations (E5). They were not used to pick the reported models.

**Selected solution.** Logistic regression is the Phase 2 champion: highest validation average precision (0.968), tied with the best test scores, best transfer to unseen areas among the fitted models, 24 interpretable coefficients, and millisecond training. Persistence and KDE stay as reference layers. Random forest and ResGCN are monitored challengers. Phase 1 crime type is shown only as low-reliability context.

---

## Main results

Test periods 11–12. Stochastic models are the mean over 10 seeds. Full tables, confidence intervals, and figures are in the report and in [`Assessment2/results/`](Assessment2/results/).

### Phase 2 — high recorded victimisation (314 cell-periods, 66 high)

| Model | Average precision | Top-20% macro-F1 |
|---|---:|---:|
| Majority (all low) | 0.210 | — |
| Persistence | 0.973 | 0.932 |
| KDE (800 m) | 0.970 | 0.932 |
| **Logistic regression** | **0.983** | **0.951** |
| RBF-SVM | 0.974 | 0.932 |
| Random forest | 0.988 | 0.951 |
| MLP | 0.980 | 0.952 |
| GCN | 0.885 | 0.860 |
| ResGCN | 0.975 | 0.929 |
| GAT | 0.867 | 0.864 |

Logistic regression under the top-20% rule: 244 true negatives, 4 false positives, 6 false negatives, 60 true positives (precision 0.938, recall 0.909). Some misses are built into the rule, because 32–35 cells can tie above the 80th percentile while the forecast flags exactly 32. Permutation tests do not separate logistic regression, random forest, the MLP, the SVM, ResGCN, and persistence. The plain GCN is significantly worse (Holm \(p = 0.010\) on one year, \(0.003\) on three years). Its precision falls once recall passes about 50%, which is the pattern expected when a high cell is blended with quieter neighbours.

Across five rolling windows, three grid sizes (1.5, 2.2, and 3.0 km), and label percentiles 70 and 80, the same order holds. At the 90th percentile the problem is harder and logistic regression no longer leads persistence. Adding neighbour averages to logistic regression does not help. Performance of the plain GCN falls as the neighbourhood widens or layers are stacked.

### Phase 1 — dominant crime type (303 cell-periods)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority (battery) | 0.366 | 0.134 |
| Persistence | 0.604 | 0.404 |
| KDE | 0.640 | 0.426 |
| **Logistic regression** | 0.611 | **0.498** |
| RBF-SVM | 0.564 | 0.467 |
| Random forest | 0.629 | 0.460 |
| MLP (validation choice) | 0.593 | 0.482 |
| GCN | 0.547 | 0.450 |
| ResGCN | 0.567 | 0.467 |
| GAT | 0.550 | 0.450 |

Battery and theft are forecast with F1 about 0.66–0.74. Criminal damage is the dominant type in only 27 test cell-periods and is usually only narrowly ahead, so the best F1 on that class is 0.27. The gain of logistic regression over persistence is suggestive (bootstrap interval on the macro-F1 difference touches values near zero; permutation \(p = 0.077\), Holm 0.53) and shrinks further in the three-year run.

### Further experiments

| ID | What it tests | Finding |
|---|---|---|
| E4 | Class weights, SMOTE, probability thresholds | Average precision barely moves. A threshold tuned on two validation periods hurts test F1. The top-32 rule avoids that instability |
| E5 | Graph radius, weights, depth, spatial lags | Message passing does not help. Constant inverse-distance weights match the unweighted graph |
| E6 | Drop feature groups | Long-run victimisation history carries Phase 2. Same-period composition is excluded, which is what made the pilot circular |
| E8 | Rolling origin, test periods 8–12 | Order of models is stable. Logistic regression leads persistence on average precision in 3 of the 5 windows |
| E9 | Five spatial blocks, labels masked | Transfer to unseen areas favours logistic regression. The graph still adds nothing |
| E10, E16 | Block bootstrap and paired permutation tests | Learned models beat the null. They do not beat persistence by a resolvable margin |
| E11, E18 | Next-period counts, then a tuned retune | Counts are forecastable. Ranking by a well-tuned count model does not overturn the classification result |
| E12 | DBSCAN hotspots and the predictive accuracy index | Useful as a micro-hotspot map. Weaker than ranked cells for area planning (PAI **1.84** versus **2.59**) |
| E13 | Runtime | A GCN fits in about 0.45 s and logistic regression in about 4 ms on CPU. Peak memory of the suite is under 1 GB |
| E14 | Validation importance | Shuffling the long-run victimisation mean drops validation average precision by 0.27 for logistic regression |
| E15 | Raw, z-score, log, log and z-score, robust scaling | The effect is modest and model-dependent. Model choice stays on the validation score of the main pipeline |
| E17 | Eighteen extra history-only Phase 1 features | Helps the random forest (macro-F1 0.518). Does not lift the neural models |
| E19 | Moran’s \(I\), lag-1 correlation, centroid graph | Effective test \(n\) is about 47–50. A centroid graph with real distance weights still does not help the GCN |
| E21 | Three years, six test periods, 12-window rolling origin | Same ranking. Average precision: random forest 0.965, ResGCN 0.964, logistic regression 0.958, persistence 0.951, GCN 0.875, GAT 0.870. Logistic regression beats persistence in 6 of 12 windows |
| E22 | FBI index violent crime only (01A, 02, 03, 04A, 04B) | Narrower label, same ranking of methods |
| E23 | Errors by community-area hardship tercile | Phase 2 recall and false-positive rate are close across terciles. Logistic regression flags 33.7% of high-hardship cell-periods and 12.5% of low-hardship ones, which tracks the recorded base rates (34.6% and 12.5%). Phase 1 is less accurate in low-hardship, low-volume, mixed cells. This is an error-disparity audit on 2008–2012 indicators, not a group-fairness audit |
| E24 | Histogram gradient boosting | Test average precision 0.975 and top-20% macro-F1 0.922. Below logistic regression on the validation rule. Phase 1 macro-F1 0.463. Extra capacity does not fix a problem set by the number of independent cells |

Figures [`Assessment2/figures/F01_pipeline.png`](Assessment2/figures/F01_pipeline.png) through [`F15_multiyear.png`](Assessment2/figures/F15_multiyear.png) are the pipeline, the calendar, exploratory plots, the grid and graph, Phase 2 comparison, ROC and precision–recall curves, confusion matrices, training curves, graph ablations, rolling origin, error maps, DBSCAN, sensitivity, the application architecture, and the three-year replication. Copies used by LaTeX live in [`Assessment2/latex/figures/`](Assessment2/latex/figures/).

---

## Repository layout

`Assessment2/code/prep.py` resolves the repository root as two directories above itself and reads the CSV from `machinelearningcode/Coding/Dataset/`. Keep this layout.

```
.
├── README.md                          this description
├── README_REPRODUCE.md                commands, environment, and experiment map
├── Assessment2/
│   ├── code/
│   │   ├── prep.py                    clean, project, grid, graph, leakage-safe features
│   │   ├── models.py                  metrics, KDE, LR, SVM, RF, MLP, GCN, ResGCN, GAT
│   │   ├── run_experiments.py         E0–E19, and E21 when A2_DATA=multi
│   │   ├── multiyear.py               three-year loader and split
│   │   ├── extensions.py              E22–E24
│   │   └── figures.py                 F01–F15
│   ├── app/
│   │   ├── score_batch.py             fit the Phase 2 logistic regression and score every cell-period
│   │   ├── server.py                  read-only JSON API and HTML dashboard
│   │   └── test_app.py                20 automated tests, parity with offline results, latency
│   ├── results/                       CSV and JSON for the one-year experiments
│   ├── results_multi/                 three-year replication
│   ├── evidence/                      terminal transcripts of the runs used in the report
│   ├── figures/                       PNG figures
│   └── latex/
│       ├── main.tex                   report source
│       ├── references.bib             APA references
│       ├── figures/                   figures embedded in the PDF
│       └── 20251674_IT8105_Assess2.pdf
└── machinelearningcode/
    ├── Coding/
    │   ├── Dataset/                   one-year CPD extract and external census files
    │   ├── Notebooks/                 Assessment 1 pilot, notebooks 01–09 (not modified)
    │   └── crime_hotspot_detection_inspect.ipynb
    └── Research Paper/                background papers and writing notes for the pilot
```

### Assessment 2 code

The pipeline starts from the raw CSV and rebuilds every intermediate table in memory. Seeds and the date split are fixed, so a rerun reproduces the tables up to floating-point differences on neural nets across hardware. Run commands from `Assessment2/code` unless a step says otherwise. The full sequence, package versions, and pitfalls are in [`README_REPRODUCE.md`](README_REPRODUCE.md).

```bash
python3 -m pip install --user numpy pandas scipy scikit-learn "imbalanced-learn==0.12.4" \
    torch torch-geometric pyproj matplotlib joblib pillow

cd Assessment2/code
python3 prep.py                    # 229,275 -> 228,595; 157 cells; 272 edges
python3 run_experiments.py         # E0–E19 on the one-year extract
python3 extensions.py              # E22–E24
python3 figures.py                 # writes ../figures/F01…F15
```

Versions used for the submitted numbers: Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1, pyproj 3.6.1, scikit-learn 1.6.1, imbalanced-learn 0.12.4, PyTorch 2.8.0, PyTorch Geometric 2.6.1, on macOS (Apple M5).

| Script | Responsibility |
|---|---|
| `prep.py` | Read the one-year extract, clean, project to EPSG:26916, build the 2.2 km grid and 3 km graph, and write the cell-period feature panel |
| `models.py` | Shared metrics, the log and z-score prepper (fit on training rows), KDE, logistic regression, SVM, random forest, MLP, GCN, ResGCN, GAT, and the training loop |
| `run_experiments.py` | Experiment registry E0–E19. `A2_DATA=multi` switches the loader, the split, and the output folder to `results_multi/` (E21). `A2_ROLL` and `A2_SUFFIX` control the long rolling origin |
| `multiyear.py` | Harmonise the 2023–2025 extract with the one-year file and print the join check |
| `extensions.py` | Violent-only label (E22), hardship-tercile error audit (E23), histogram gradient boosting (E24) |
| `figures.py` | All report figures. Run after the experiments whose tables the figures read |

### Report

[`Assessment2/latex/main.tex`](Assessment2/latex/main.tex) is the final report: abstract, introduction, literature review, methodology, results and discussion (E0–E24), application design, conclusion, and appendices (data dictionary, split manifest, reproduction, learning outcomes, traceability to Assessment 1, evidence gaps, and run transcripts).

```bash
cd Assessment2/latex
tectonic main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Assessment 1 pilot (`machinelearningcode`)

These notebooks are the proposal pipeline. They are the evidence that E0 audits. They are not the source of the Assessment 2 tables.

| Notebook | What the pilot did |
|---|---|
| [`01_data_inspection.ipynb`](machinelearningcode/Coding/Notebooks/01_data_inspection.ipynb) | Schema, types, and missingness of the one-year extract |
| [`02_data_cleaning.ipynb`](machinelearningcode/Coding/Notebooks/02_data_cleaning.ipynb) | Duplicates, coordinates, and the study boundary |
| [`03_eda.ipynb`](machinelearningcode/Coding/Notebooks/03_eda.ipynb) | Volumes, categories, and concentration |
| [`04_spatial_grid.ipynb`](machinelearningcode/Coding/Notebooks/04_spatial_grid.ipynb) | 2.2 km grid and occupied cells |
| [`05_graph_construction.ipynb`](machinelearningcode/Coding/Notebooks/05_graph_construction.ipynb) | 3 km graph on cell centres |
| [`06_temporal_crime_features.ipynb`](machinelearningcode/Coding/Notebooks/06_temporal_crime_features.ipynb) | Phase 1 node features and the dominant-type label |
| [`07_model_spatial_comparison.ipynb`](machinelearningcode/Coding/Notebooks/07_model_spatial_comparison.ipynb) | Phase 1 GCN against SVM and KDE |
| [`08_victimisation_risk_features.ipynb`](machinelearningcode/Coding/Notebooks/08_victimisation_risk_features.ipynb) | Phase 2 victimisation features and the high-cell label |
| [`09_final_predictions_and_evaluation.ipynb`](machinelearningcode/Coding/Notebooks/09_final_predictions_and_evaluation.ipynb) | Pilot scores written to `Notebooks/results/` |
| [`crime_hotspot_detection_inspect.ipynb`](machinelearningcode/Coding/crime_hotspot_detection_inspect.ipynb) | Early inspection notebook |

Saved pilot outputs live under [`machinelearningcode/Coding/Notebooks/data/`](machinelearningcode/Coding/Notebooks/data/), [`models/`](machinelearningcode/Coding/Notebooks/models/), and [`results/`](machinelearningcode/Coding/Notebooks/results/). The background readings are in [`machinelearningcode/Research Paper/`](machinelearningcode/Research%20Paper/).

---

## Prototype application

[`Assessment2/app/`](Assessment2/app/) is a tested prototype of the design in the report. It is not deployed.

The intended users are analysts building a 28-day prevention plan (outreach, environmental design, victim support, patrol). After a period closes, a batch job refits the logistic regression on history-only features, ranks every cell, and stores bootstrap rank intervals and the top coefficients. The analyst sees the top 32 cells, the change in rank, a 13-period sparkline, and a near-cutoff tag. Cells within two ranks of the cutoff hold a disproportionate share of the test errors (4 of 10 errors sit in 10 of 314 cell-periods). The analyst logs include, exclude, or review, with a written reason. The model never allocates resources.

```bash
cd Assessment2/app
python3 score_batch.py          # writes artifacts/; matches the E2 logistic regression
python3 test_app.py             # 20 tests; results/E20_prototype_test.json
A2_KEYS="mykey:analyst" python3 server.py --port 8080
curl -H "X-API-Key: mykey" "http://127.0.0.1:8080/v1/forecasts?period=13&top_k=32"
curl -H "X-API-Key: mykey" "http://127.0.0.1:8080/dashboard?period=13" > dashboard.html
```

The API is read-only and uses the Python standard library. Endpoints are `/v1/health`, `/v1/forecasts`, `/v1/cells/<grid_id>/history` (analyst role), and `/dashboard`. A missing or wrong key returns 401. A viewer cannot read cell history (403). Unknown cells, bad periods, and an unknown grid version are rejected. At start-up the server refuses a prediction store whose feature-schema hash does not match the code. Every request is appended to an audit log. Responses carry the disclaimer that these are area-level recorded-incident forecasts.

All **20** automated tests passed. Served forecasts for the two test periods match the offline E2 file exactly (average precision 0.983, top-20% precision 0.938, recall 0.909). Median latency on 600 local requests was 0.40 ms. That figure is a same-machine loop. It says nothing about a network under load. Online inference, single sign-on, automated retraining, drift detectors, and a fairness review are specified in the report and are not implemented.

---

## Limitations

- One city (Chicago), 2023–2026, a 2.2 km grid, and 28-day periods. Transfer to New Zealand cities is untested. The course context started from New Zealand, and no equivalent public incident file with coordinates was available.
- About 157 cells, with temporal correlation near 0.96 and Moran’s \(I\) near 0.5. Extra periods add little independent information. Differences of about 0.01 average precision cannot be resolved.
- Recorded crime mixes where harm occurred with where it was reported and policed. A deployment that sends more enforcement to flagged cells can raise later recorded counts and keep those cells at the top of the list. No demographic attribute is a model input. The E23 audit is the bound on what this repository claims about disparity.
- The main test window is one summer. The three-year replication and the 12-window rolling origin cover more seasons in the same city.
- Victimisation is an incident count. Homicides in the source are counted per victim, which is slightly inconsistent with the other categories. The Loop and other high-footfall cells score high because many people pass through them.
- Land use, population, points of interest, weather, and events are absent, so the model cannot separate activity from reporting.
- Experiments E15–E19 and E22–E24 were designed after the main test results were known. Their choices use validation data only and do not change the selected models. They are post-hoc.
- The reporting-bias and feedback-loop problems have a governance design (human review, delayed-label monitoring, a champion–challenger gate, quarterly comparison with an independent harm measure) and no operational test.

---

## Licence and data terms

Coursework by Samina Tuladhar (20251674), Whitecliffe, IT8105, 2026. The crime extracts are City of Chicago open data, supplied “as is” under the City’s terms of use. Any application built on them must say that the data have been modified from [www.cityofchicago.org](https://www.cityofchicago.org). This repository stores block-level coordinates and area aggregates for non-commercial research. It does not store names or narratives. Test API keys are generated at run time and are not committed.
