# Demystifying the Recency Heuristic


## Reproducing the plots from the paper

First, generate experiment data using

```bash
python scripts/automate.py --go
python scripts/consolidate.py results/ --clean
```

Then, call this script to plot the figures:

```bash
python plot_all_paper_figures.py
```
