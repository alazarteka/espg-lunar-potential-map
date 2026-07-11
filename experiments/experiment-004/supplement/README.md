# Experiment 004 Supplement

`main.tex` is the tracked scientific account of Phases A1-A4. It records the
methods, frozen comparisons, results, protocol corrections, and final decision
without replacing the machine-readable CSV artifacts.

Compile from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The tracked `main.pdf` was verified with Tectonic 0.15.0 when `pdflatex` was not
available in the experiment environment:

```bash
tectonic --keep-logs main.tex
```
