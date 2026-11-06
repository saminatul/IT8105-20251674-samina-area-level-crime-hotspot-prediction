# Area-level crime hotspot and high-victimisation risk prediction

IT8105 Machine Learning, Assessment 2. Samina Tuladhar (20251674), Whitecliffe.

The report is `Assessment2/latex/main.pdf`. The source is `Assessment2/latex/main.tex` and `Assessment2/latex/references.bib`. Reproduction steps are in `Assessment2/README_REPRODUCE.md`.

The study forecasts, for each 2.2 km cell and the next 28 days:

1. the dominant recorded crime type;
2. whether the cell is in the top 20% for recorded person-directed incidents.

Features use only earlier periods. The selected Phase 2 model is logistic regression. A persistence rule is nearly as accurate. The pilot graph convolutional network does not beat that baseline. Kernel density estimation and DBSCAN are the spatial benchmarks. Results are area-level recorded incidents, not unique victims and not an individual risk.

## Layout

Keep this layout. `Assessment2/code/prep.py` resolves the data file from the repository root.

