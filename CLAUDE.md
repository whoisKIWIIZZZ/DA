# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Source-Free Domain Adaptation (SFDA) research codebase implementing multiple methods including SHOT, NRC, COWA, DIFO, ProDe, CausalDA, and others. Works with datasets: Office-31, Office-Home, VISDA-C, DomainNet-126, and ImageNet variants (A, R, V2, Sketch).


## Entry Points

Three main scripts, chosen based on the method's backbone and dataset:

- **`image_target_of_oh_vs.py`** — Office-Home, Office-31, VISDA-C. Methods live in `src/methods/oh/`. Supports: nrc, shot, sclm, cowa, gkd, tpds, CausalDA, difo, ProDe, tsd, plue, adacontrast, source.
- **`image_target_of_oh_vs_new.py`** — Extended version of the above. Adds: difo_plus, difo_plus_partial, difo_plus_open, ProDe_diff, ProDe_partial, ProDe_open, diff_align, lcfd, source_clip, source_neg, diffadd, draws.
- **`image_target_in_126.py`** — DomainNet-126 and ImageNet variants (A, R, V, K). Methods live in `src/methods/net/`. Supports: nrc, shot, sclm, cowa, gkd, tpds, CausalDA, difo, ProDe, tsd, plue, adacontrast, source, diff_new_opt_new, diff_align_plus, difo_plus.

Shell scripts in the root (`office_home.sh`, `office31.sh`, `domainet126.sh`, `visda_c.sh`) wrap the training commands with domain index iteration.
