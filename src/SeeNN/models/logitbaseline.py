"""
non-deep baseline: HOG features + class-weighted multinomial logistic regression per milestone 2

Trains and tunes the model on CPU (~7 minutes on a 7950x3d) if no prediction file exists for the current 
settings, otherwise it reuses saved predictions. Delete the predictions file to retrain.

Each 248×248 scan is resized to 124×124 and converted into 7,056 HOG features (9 orientations, 8×8 cells, 2×2 blocks). 
Features are standardized using training data only, classes are weighted inversely by frequency, 
and regularization strength (C) is tuned on validation data. The process is deterministic.

Saves validation predictions (four class probabilities per scan) and the C-value tuning results to results/logitbaseline/, 
with settings and seed encoded in the prediction filename. Replaces predictions from previous settings to avoid 
treating them as separate runs. Saves the fitted scaler and model to checkpoints/logitbaseline.joblib (excluded from git via .gitignore).

Finally, runs evaluatebaseline.py to generate the results table and figures
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/SeeNN, for the modules below
import evaluatebaseline
from split import CLASSES, SEED # from previous milestone
from standardize import OUTPUT_FOLDER

RESULTS = evaluatebaseline.RESULTS_FOLDER / "logitbaseline"
CHECKPOINT = evaluatebaseline.ROOT / "checkpoints" / "logitbaseline.joblib"
SETTINGS = {"seed": SEED, "hog_size": 124, "orientations": 9, "pixels_per_cell": 8, "cells_per_block": 2,
            "C_grid": [0.0001, 0.001, 0.01, 0.1], "class_weight": "balanced", "max_iter": 3000}
# every setting above goes into the predictions file's name, so a file is only reused by the settings that made it
TAG = (f"seed{SETTINGS['seed']}_hog{SETTINGS['hog_size']}_orient{SETTINGS['orientations']}"
       f"_cell{SETTINGS['pixels_per_cell']}_block{SETTINGS['cells_per_block']}"
       f"_C{'-'.join(f'{c:g}' for c in SETTINGS['C_grid'])}_{SETTINGS['class_weight']}_iter{SETTINGS['max_iter']}")
PREDICTIONS = RESULTS / f"predictions_val_{TAG}.csv"
WORKERS = max(1, (getattr(os, "process_cpu_count", os.cpu_count)() or 1) - 2)
CHUNK_SIZE = 256  # scans featurized per worker task


def hog_features(output):
    """HOG vector of one standardized PNG after resizing to hog_size"""
    size = SETTINGS["hog_size"]
    with Image.open(OUTPUT_FOLDER / output) as image:
        pixels = np.asarray(image.resize((size, size), Image.Resampling.BOX), dtype=np.float32) / 255
    return hog(pixels, orientations=SETTINGS["orientations"], pixels_per_cell=(SETTINGS["pixels_per_cell"],) * 2,
               cells_per_block=(SETTINGS["cells_per_block"],) * 2, feature_vector=True).astype(np.float32)


def hog_chunk(outputs):
    """runs in a worker process so each task returns one array instead of one per scan"""
    return np.stack([hog_features(output) for output in outputs])


def featurize(rows):
    outputs = rows["output"].tolist()
    chunks = [outputs[i:i + CHUNK_SIZE] for i in range(0, len(outputs), CHUNK_SIZE)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return np.concatenate(list(pool.map(hog_chunk, chunks)))  # map to retain original order


def weighted_log_loss(y, prob):
    """log loss with each class weighted equally (the same weighting as the training loss)"""
    return log_loss(y, prob, sample_weight=compute_sample_weight("balanced", y), labels=list(range(len(CLASSES))))


def train(train_rows, val_rows):
    """fits one model per C on train, keeps the one with the lowest weighted val log loss"""
    started = time.perf_counter()
    x_train, x_val = featurize(train_rows), featurize(val_rows)
    print(f"HOG: {x_train.shape[1]:,} features per scan ({time.perf_counter() - started:.0f} s)", flush=True)
    scaler = StandardScaler().fit(x_train)  # training statistics only
    x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
    y_train = train_rows["class"].map(CLASSES.index).to_numpy()
    y_val = val_rows["class"].map(CLASSES.index).to_numpy()

    sweep, models = [], []
    for c in SETTINGS["C_grid"]:
        model = LogisticRegression(C=c, class_weight=SETTINGS["class_weight"], max_iter=SETTINGS["max_iter"])
        model.fit(x_train, y_train)
        prob_val = model.predict_proba(x_val)
        sweep.append({"C": c, "train_weighted_log_loss": weighted_log_loss(y_train, model.predict_proba(x_train)),
                      "val_weighted_log_loss": weighted_log_loss(y_val, prob_val),
                      "val_macro_f1": evaluatebaseline.macro_f1(evaluatebaseline.confusion(y_val, prob_val.argmax(axis=1))),
                      "iterations": int(model.n_iter_.max())})
        models.append(model)
        print(f"  C={c:g}: val weighted log loss {sweep[-1]['val_weighted_log_loss']:.4f}, "
              f"val macro F1 {sweep[-1]['val_macro_f1']:.4f}", flush=True)
    best = int(np.argmin([s["val_weighted_log_loss"] for s in sweep]))
    sweep = pd.DataFrame(sweep)
    sweep["chosen"] = sweep.index == best # mark the best C in the sweep DataFrame
    return {"scaler": scaler, "model": models[best]}, sweep # return the fitted scaler, and the model for the best C alongside the sweep DataFrame


def save_predictions(val_rows, prob):
    predictions = val_rows[["file", "patient", "group", "class", "size"]].copy()
    predictions["pred"] = [CLASSES[k] for k in prob.argmax(axis=1)]
    for k, c in enumerate(CLASSES):
        predictions[f"p_{c}"] = prob[:, k].round(6)
    for old in RESULTS.glob("predictions_val*.csv"):  # from other settings
        old.unlink()
    predictions.to_csv(PREDICTIONS, index=False)


def main():
    name = PREDICTIONS.relative_to(evaluatebaseline.ROOT).as_posix()
    if PREDICTIONS.exists():
        print(f"reusing {name}\n(delete PREDICTIONS or change SETTINGS to retrain)\n")
    else:
        print(f"no {name} yet, training from scratch", flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        val_rows = evaluatebaseline.read_split("val")
        fitted, sweep = train(evaluatebaseline.read_split("train"), val_rows)
        sweep.round(4).to_csv(RESULTS / "sweep.csv", index=False)
        CHECKPOINT.parent.mkdir(exist_ok=True)
        joblib.dump(fitted, CHECKPOINT)
        print(f"  chose C={sweep.loc[sweep['chosen'], 'C'].iloc[0]:g}, "
              f"saved {CHECKPOINT.relative_to(evaluatebaseline.ROOT).as_posix()}")
        prob = fitted["model"].predict_proba(fitted["scaler"].transform(featurize(val_rows)))
        save_predictions(val_rows, prob)  # outputs the predictions file 
        # this is done at the end so that any half-completed run that doesn't get here will be retrained on the next run
        print()
    evaluatebaseline.main()


if __name__ == "__main__":
    main()