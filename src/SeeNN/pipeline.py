"""
SCRIPTS FOR LOADING INTO MODULES, THIS PIPELINE ISN'T CONFIGURED FOR DIRECT EXECUTION
import it into other scripts so that they can call the functions in here and load our cleaned data!!!
currently configured for tf, but might swap it to keras -> pytorch later bc tf doesn't support cuda on native windows atm
but tbh tf is fine for now on cpu because of how light the data are

tf.data input pipeline for the standardized OCT images, the same for every split

each example is one 248x248 grayscale PNG from standardize.py, 
with pixel values normalized using the mean and std of all pixels in the training split, 
and an integer label 0-3 in the order of CLASSES. the same normalization stats are used 
for every split, and only the training split is shuffled

run hash.py, dupe.py, standardize.py, and split.py first. the normalization stats 
are computed once and saved to norm_stats.csv, then reused unless the number of training images changes
"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from PIL import Image

from split import CLASSES, DATA_FOLDER, SEED, SPLITS_CSV
from standardize import OUTPUT_FOLDER, OUTPUT_HEIGHT, OUTPUT_WIDTH

STATS_CSV = DATA_FOLDER / "norm_stats.csv"
INPUT_SHAPE = (OUTPUT_HEIGHT, OUTPUT_WIDTH, 1)
BATCH_SIZE = 64
THREADS = max(1, (getattr(os, "process_cpu_count", os.cpu_count)() or 1) - 2)


def read_split(name):
    """returns (png paths, integer labels) for one split in splits.csv"""
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == name]
    paths = [(OUTPUT_FOLDER / row["output"]).as_posix() for row in rows]
    labels = [CLASSES.index(row["class"]) for row in rows]
    return paths, labels


def pixel_histograms(paths):
    """256 bin histogram of every image, one row per path"""
    def one(path):
        with Image.open(path) as image:
            return image.histogram()
    with ThreadPoolExecutor(THREADS) as pool:
        return np.array(list(pool.map(one, paths)), dtype=np.int64)


def train_stats():
    """pixel mean and std over the whole training split
    saved to STATS_CSV so every split is scaled by the same numbers"""
    paths, _ = read_split("train")
    hist = pixel_histograms(paths).sum(axis=0)
    levels = np.arange(256) # domain of pixel values
    mean = (hist * levels).sum() / hist.sum() # train pop mean
    std = np.sqrt((hist * (levels - mean) ** 2).sum() / hist.sum()) # again, pop mean instead of sample mean, since we're normalizing the whole population
    stats = {"mean": float(mean), "std": float(std), "train_images": len(paths)}
    with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats))
        writer.writeheader()
        writer.writerow(stats)
    return stats


def load_stats():
    """reads STATS_CSV, recomputing it if missing or if the training split has changed size"""
    if STATS_CSV.exists():
        with open(STATS_CSV, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        stats = {"mean": float(row["mean"]), "std": float(row["std"]), "train_images": int(row["train_images"])}
        if stats["train_images"] == len(read_split("train")[0]): # if the number of training images has not changed, return the cached stats
            return stats
    return train_stats()


def decode(path, label):
    """reads one image as a 248x248x1 tensor"""
    image = tf.io.decode_png(tf.io.read_file(path), channels=1) # decode as grayscale
    image.set_shape(INPUT_SHAPE)
    return image, label


def make_dataset(name, batch_size=BATCH_SIZE, shuffle=None):
    """batches of (float32 images, int32 labels) for one split.
    decode in parallel -> cache the uint8 images in memory -> 
    shuffle (train only) -> batch -> normalize -> prefetch"""
    stats = load_stats()
    mean, std = stats["mean"], stats["std"]
    if shuffle is None:
        shuffle = name == "train"
    paths, labels = read_split(name) # read the split from splits.csv, only the images that made it through standardize.py are included

    def normalize(images, labels):
        return (tf.cast(images, tf.float32) - mean) / std, labels # normalize to zero mean and unit std using the training split stats

    ds = tf.data.Dataset.from_tensor_slices((paths, tf.constant(labels, tf.int32))) # create a dataset of (path, label) pairs
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).cache() # decode the PNGs in parallel and cache the uint8 images in memory
    if shuffle:
        # the files are sorted by class, so the buffer has to hold the whole split to mix them
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True) # shuffle the training split, but not the validation or test splits
    ds = ds.batch(batch_size).map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def denormalize(images):
    """undoes normalize, back to 0-255 for display"""
    stats = load_stats()
    return np.clip(np.asarray(images) * stats["std"] + stats["mean"], 0, 255)