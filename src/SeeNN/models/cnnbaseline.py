"""
simple deep baseline: small CNN trained from scratch (section 5 of the proposal)

Architecture: four blocks of two 3x3 conv layers (32, 64, 128, 256 filters), each with batch normalization and relu, and max 
pooling after each block, then global average pooling, dropout and a softmax over the 4 classes (1,174,628 trainable params).

Trained with adam and class-weighted cross-entropy - same class weights as the logistic regression baseline, so both
are trained for macro F1. Training stops once val loss hasn't improved for 3 epochs, and the best epoch is kept.

Run the learning check in notebooks/02_baselines.ipynb first (trains on 16 scans until the model fits them, to make sure
it can learn). This script then does full training for each of the 3 seeds (each seed changes the starting weights and
the shuffle order). Settings and seeds are in configs/cnnbaseline.yaml.

Saves per-epoch history and val predictions to results/cnnbaseline/, and the trained models to checkpoints/ (not in git).
Keras 3 runs the model on PyTorch, which can use an NVIDIA GPU on native Windows (TensorFlow can't after 2.10).
TensorFlow's tf.data (pipeline.py) still loads and batches the images, on the CPU.

Runs evaluatebaseline.py at the end for the results table and figures.
"""
import os

# both have to be set before keras and torch are imported
os.environ["KERAS_BACKEND"] = "torch"  # Keras 3 runs the model on PyTorch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # lets cuBLAS run deterministically (see main)

import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf  # only for the tf.data input pipeline
import torch
from keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

if keras.backend.backend() != "torch":  # keras was already imported with another backend, e.g. earlier in a notebook
    raise RuntimeError("set KERAS_BACKEND=torch before importing keras")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/SeeNN, for the modules below
import evaluatebaseline
import pipeline
from config import CHECKPOINTS_FOLDER, run_config
from split import CLASSES  # from previous milestone

RESULTS = evaluatebaseline.RESULTS_FOLDER / "cnnbaseline"
RUN = run_config("cnnbaseline")  # configs/cnnbaseline.yaml
SEEDS = RUN["seeds"]
SETTINGS = {name: RUN[name] for name in ("dropout", "learning_rate", "max_epochs", "patience")}
SETTINGS["batch_size"] = pipeline.BATCH_SIZE  # from config.yaml
# every setting above goes into the predictions file's name, so a file is only reused by the settings that made it
TAG = (f"in{pipeline.INPUT_SHAPE[0]}_drop{SETTINGS['dropout']:g}_adam{SETTINGS['learning_rate']:g}"
       f"_batch{SETTINGS['batch_size']}_ep{SETTINGS['max_epochs']}_pat{SETTINGS['patience']}")


def predictions_path(seed):
    return RESULTS / f"predictions_val_seed{seed}_{TAG}.csv"


def history_path(seed):
    return RESULTS / f"history_seed{seed}_{TAG}.csv"


def build_model(dropout):
    """four conv blocks -> global average pooling -> dropout -> softmax, compiled with Adam and cross-entropy"""
    # could use for loop, but for sake of clear visualization, wrote out each of the blocks
    model = models.Sequential([
        layers.Input(shape=pipeline.INPUT_SHAPE),

        # block 1: 248 x 248 x 32
        layers.Conv2D(32, (3, 3), padding="same"), # add padding to keep edges
        layers.BatchNormalization(), # normalize to keep training stable - recommended in past studies
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # block 2: 124 x 124 x 64
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # block 3: 62 x 62 x 128
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # block 4: 31 x 31 x 256
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(), # 256 numbers (one per feature map) instead of flattening
        layers.Dropout(dropout), # dropout to help with overfitting
        layers.Dense(len(CLASSES), activation="softmax") # probabilities of CNV, DME, DRUSEN, NORMAL
    ])
    model.compile(loss="sparse_categorical_crossentropy", # sparse for class numbers 0-3 (not one-hot vectors)
                  optimizer=keras.optimizers.Adam(learning_rate=SETTINGS["learning_rate"]), # as learned in class
                  metrics=["accuracy"])
    return model


def add_weights(ds, labels):
    """adds a weight to every scan so each class counts equally in the loss. weights come from the split's own class
    counts and average 1 over the split (the same weighting as weighted_log_loss in logitbaseline.py)"""
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=labels)  # one weight per class
    weights = tf.constant(weights, tf.float32)

    def add_weight(images, y):
        return images, y, tf.gather(weights, y)  # each scan gets its class's weight

    return ds.map(add_weight)


def train(seed):
    """trains one model on train, keeping the weights from the epoch with the lowest class-weighted val loss"""
    keras.utils.set_random_seed(seed)  # python, numpy, tensorflow and torch: initial weights and dropout
    _, train_labels = pipeline.read_split("train")
    _, val_labels = pipeline.read_split("val")
    train_ds = add_weights(pipeline.make_dataset("train", SETTINGS["batch_size"], seed=seed), train_labels) # shuffle with the current seed so every seed has different order
    val_ds = pipeline.make_dataset("val", SETTINGS["batch_size"])  # not shuffled, in splits.csv order

    model = build_model(SETTINGS["dropout"])

    # stop training once it's no longer improving - save weights from eopch with lowest validation loss
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=SETTINGS["patience"], restore_best_weights=True)
    started = time.perf_counter() # start timer to measure training time
    fit = model.fit(train_ds, validation_data=add_weights(val_ds, val_labels), epochs=SETTINGS["max_epochs"],
                    callbacks=[stop], verbose=2,  # verbose=2 one line per epoch for progress
                    shuffle=False)  # tf.data already shuffles train; this stops keras warning that it can't

    # save training results per epoch (used for loss-curve figure)
    history = pd.DataFrame(fit.history)
    history.insert(0, "epoch", range(1, len(history) + 1))
    history["train_minutes"] = (time.perf_counter() - started) / 60  # total for the whole run, on every row
    history["chosen"] = history.index == history["val_loss"].idxmin()  # the epoch whose weights were kept
    prob = model.predict(val_ds, verbose=0)
    return model, history, prob


def save_predictions(val_rows, prob, seed):
    """turns trained model's raw output into CSV file for evaluatebaseline script"""
    predictions = val_rows[["file", "patient", "group", "class", "size"]].copy()
    predictions["pred"] = [CLASSES[k] for k in prob.argmax(axis=1)]
    for k, c in enumerate(CLASSES):
        predictions[f"p_{c}"] = prob[:, k].round(6)
    predictions.to_csv(predictions_path(seed), index=False)


def remove_other_settings():
    """deletes predictions and histories from other settings, so they're not mixed in with this run's seeds"""
    keep = {predictions_path(seed) for seed in SEEDS} | {history_path(seed) for seed in SEEDS}
    for old in [*RESULTS.glob("predictions_val*.csv"), *RESULTS.glob("history_seed*.csv")]:
        if old not in keep:
            print(f"  removing {old.name} (other settings)")
            old.unlink()


def main():
    torch.use_deterministic_algorithms(True)  # same seed, same run: deterministic GPU kernels, or an error naming the op
    RESULTS.mkdir(parents=True, exist_ok=True)
    remove_other_settings()
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none found, training on the CPU'}\n")

    # full training, once per seed
    val_rows = evaluatebaseline.read_split("val")
    for seed in SEEDS:
        path = predictions_path(seed)
        if path.exists():
            print(f"reusing {path.name}")
            continue
        print(f"no {path.name} yet, training seed {seed} from scratch", flush=True)
        model, history, prob = train(seed)
        history.round(4).to_csv(history_path(seed), index=False)
        CHECKPOINTS_FOLDER.mkdir(exist_ok=True)
        model.save(CHECKPOINTS_FOLDER / f"cnnbaseline_seed{seed}.keras") # save layers and weights so model can be reloaded without retraining
        save_predictions(val_rows, prob, seed)  # last, so a half-finished run is retrained next time
    evaluatebaseline.main()


if __name__ == "__main__":
    main()
