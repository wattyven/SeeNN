## run the scripts in the following order:

hash.py -> dupe.py -> standardize.py -> split.py

notebooks/01_eda.ipynb can be run afterwards. it relies on the output of the 4 scripts above as well as the helper functions in pipeline.py

settings shared by every script (paths, seed, classes, val fraction, resolution, batch size) are in config.yaml at the repo root, read through config.py
