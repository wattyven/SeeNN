# SeeNN

A CNN triage tool for retinal OCT scans, sorting B-scans into an urgent (CNV or DME), routine (drusen) or observation (normal) referral, for clinics that have an OCT machine but no retina specialist on site. NOT EXAMINED BY MEDICAL PROFESSIONALS.

## Data

Kermany, Zhang and Goldbaum, *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images*, Mendeley Data, **Version 3** ([doi:10.17632/rscbjbr9sj.3](https://data.mendeley.com/datasets/rscbjbr9sj/3)), downloaded 26 September 2026, licensed CC BY 4.0.

Download it, unzip the nested archives, and move ONLY the `OCT` folder (containing `train/` and `test/`) to `data/OCT/`.

## Setup

Python 3.14 (tested on 3.14.8) on Windows 11. Everything so far through M3 runs solely on CPU; TensorFlow (tested ver. 2.22) has no GPU support on native Windows after 2.10 without using WSL2.

```
pip install -r requirements.txt
pip install --force-reinstall --no-deps keras==3.15.1
```

The second command is needed because tensorflow 2.22.0rc0 also installs keras-nightly into the same folder.

Shared settings (paths, seed, classes, val fraction, resolution, batch size) are in `config.yaml`, and each baseline's settings and seeds are in `configs/`. Every script and notebook reads them through `src/SeeNN/config.py`.

## Reproduce

From the repository root, in order:

```
python src/SeeNN/hash.py          # SHA-256 of every image and its mirror -> data/OCT/image_hashes.csv
python src/SeeNN/dupe.py          # duplicates and conflicting labels to drop -> data/OCT/omit_list.csv
python src/SeeNN/standardize.py   # grayscale, wedge fill, 496-row crop, 248x248 PNGs -> data/OCT_standardized/
python src/SeeNN/split.py         # patient-grouped train/val split -> data/OCT/splits.csv
python -m pytest tests            # no patient, group or image in more than one set
```

Then run `notebooks/01_eda.ipynb` from top to bottom for the figures, normalization statistics, pipeline timings and smoke test.

## Baselines (M4)

Every model is scored the same way on the validation split, with macro F1 as the primary metric, and without touching the test set.

```
python src/SeeNN/models/logitbaseline.py  # HOG + logistic regression, trained if no saved predictions in destination folder
python src/SeeNN/evaluatebaseline.py      # results table and figures for every model from saved predictions
```

## Structure

```
config.yaml             shared settings for every script and notebook
configs/                settings and seeds for each baseline
notebooks/              01_eda, 03_baselines
src/SeeNN/
  config.py             reads config.yaml and configs/
  evaluatebaseline.py
  models/
results/
tests/
data/                   .gitignore
checkpoints/            .gitignore

```

## License

Code: Apache-2.0 (see `LICENSE`). Data: CC BY 4.0.
