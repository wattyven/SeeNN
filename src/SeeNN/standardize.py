"""
cleans, crops, and resizes OCT images for input to our nn, in two steps

1. treat every image that is NOT omitted by cropping it to CROP_HEIGHT rows, saving results as PNGs
2. write the log and print a summary
"""

import csv
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from PIL import Image, ImageDraw


DATA_FOLDER = Path(__file__).resolve().parents[2] / "data" / "OCT"
OUTPUT_FOLDER = DATA_FOLDER.parent / "OCT_standardized"
OMIT_CSV = DATA_FOLDER / "omit_list.csv"
LOG_CSV = DATA_FOLDER / "standardize_log.csv"

CROP_HEIGHT = 496 # height of almost every scan in the dataset, so most scans aren't cropped at all
OUTPUT_WIDTH, OUTPUT_HEIGHT = 248, 248 # half of CROP_HEIGHT, so the vertical resize is an even 2:1

WORKERS = max(1, (getattr(os, "process_cpu_count", os.cpu_count)() or 1) - 2)  # 1 for no multiprocessing
CHUNK_SIZE = 32 # number of images handed to a worker for processing at a time

SPLITS = ["train", "test"]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


# finding images

def find_image_files(splits):
    """returns (split, class, path) for each jpeg"""
    files = []
    for split in splits:
        for class_name in CLASSES:
            folder = DATA_FOLDER / split / class_name
            if not folder.is_dir():
                print(f"Warning: folder not found: {folder}")
                continue
            files += [(split, class_name, path) for path in sorted(folder.iterdir())
                      if path.suffix.lower() in (".jpg", ".jpeg")]
    return files


def relative_name(path):
    return path.relative_to(DATA_FOLDER).as_posix()


# for a single image

def load_and_clean(path):
    """for a single image, load, convert to grayscale, and fill in empty white corners with black pixels"""
    with Image.open(path) as image:
        gray = image.convert("L")
    w, h = gray.size
    corners_removed = False
    # rotated scans have white wedges in the corners that aren't part of the scan,
    # so whenever we find a triangle of white pixels in a corner, fill in the area up to 15 pixel values away 
    # with black pixels (this is a really hacky way to avoid rotating images or doing other transforms)
    for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if gray.getpixel(corner) >= 240:
            ImageDraw.floodfill(gray, corner, 0, thresh=15)
            corners_removed = True
    return gray, corners_removed


def standardize(path):
    """standardizes one image and returns (cleaned image, result, details)"""
    gray, corners_removed = load_and_clean(path)
    # keep the top CROP_HEIGHT rows, which leaves 496 px scans as is, trims the bottom 16 rows off
    # 512 px scans (their retinas sit near the top), and pads the couple of shorter scans with black
    # always keeps the full width and resizes to a square
    # wider scans get compressed more as a result, but the vertical scale remains consistent
    result = gray.crop((0, 0, gray.width, CROP_HEIGHT)).resize(
        (OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
    info = {
        "width": gray.width, "height": gray.height,
        "white_corners_removed": "yes" if corners_removed else "no",
    }
    return gray, result, info


def preview(path):
    """previews what a crop does to a single image with a side by side comparison.
    the red box is the crop. returns the image to show or save"""
    gray, result, _ = standardize(path)
    canvas = Image.new("RGB", (gray.width + 20 + result.width, max(gray.height, result.height)), "white")
    canvas.paste(gray.convert("RGB"), (0, 0))
    canvas.paste(result.convert("RGB"), (gray.width + 20, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, gray.width - 1, CROP_HEIGHT - 1],
                   outline=(255, 0, 0), width=3)
    return canvas


# for worker processes for parallelization

def process_one_image(task):
    """it's kind of in the name, huh. returns (log row, None), or (None, error) if the image couldn't be read"""
    split, class_name, path = task
    try:
        _, result, info = standardize(path)
    except Exception as error:  # skip damaged or unreadable images, but report them in error output
        return None, f"{relative_name(path)}: {error}"
    out_path = OUTPUT_FOLDER / split / class_name / f"{path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    output = out_path.relative_to(OUTPUT_FOLDER).as_posix()
    return {"file": relative_name(path), "output": output, **info}, None


# steps for standardizing images

def process_all(omit, run):
    """standardize all images that are not omitted, return the log rows and counts for the report"""
    files = find_image_files(SPLITS)
    tasks = [task for task in files if relative_name(task[2]) not in omit]
    counts = Counter(skipped=len(files) - len(tasks))
    print(f"Processing {len(tasks):,} images")

    # old log is for old runs, so delete it to avoid confusion
    LOG_CSV.unlink(missing_ok=True)
    log_rows = []
    started = time.perf_counter()
    results = run(process_one_image, tasks)
    for done, (row, error) in enumerate(results, start=1):
        if error:
            print(f"  Could not process: {error}")
            counts["failed"] += 1
            continue
        log_rows.append(row)
        if row["white_corners_removed"] == "yes":
            counts["corners"] += 1
        if done % 5000 == 0:
            print(f"  {done:,} of {len(tasks):,} done ({time.perf_counter() - started:.0f} s)")
    return log_rows, counts


def report(log_rows, counts):
    """write the log and generate a report of the run, including a summary of the crop height and counts of images processed, skipped, and failed"""
    print("\nsummary of changes")
    if not log_rows:
        print("no images saved")
        return
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)

    def show(label, value):
        print(f"  {label + ':':<32}{value}")

    show("Crop height", f"{CROP_HEIGHT} rows")
    show("Saved", f"{len(log_rows):,} images at {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    show("Skipped", f"{counts['skipped']:,}")
    show("Could not be processed", f"{counts['failed']:,}")
    show("White corners removed", f"{counts['corners']:,}")
    show("Image log", LOG_CSV)


def check_setup():
    """fancy assert, returns true if we're good to go, otherwise prints a message and returns false"""
    problems = [
        (not DATA_FOLDER.is_dir(), f"DATA_FOLDER does not exist: {DATA_FOLDER}"),
        (not OMIT_CSV.exists(), f"{OMIT_CSV.name} not found. Run dupe.py first."),
        (not isinstance(WORKERS, int) or WORKERS < 1,
         "WORKERS must be a whole number of at least 1."),
        (OUTPUT_FOLDER.resolve().is_relative_to(DATA_FOLDER.resolve()),
         "OUTPUT_FOLDER must not be inside DATA_FOLDER."),
        (OUTPUT_FOLDER.exists() and any(OUTPUT_FOLDER.iterdir()),
         f"{OUTPUT_FOLDER} is not empty.\nDelete or rename it before running again."),
    ]
    for failed, message in problems:
        if failed:
            print(message)
            return False
    return True


def main():
    if not check_setup():
        return
    started = time.perf_counter()
    with open(OMIT_CSV, newline="", encoding="utf-8") as file:
        omit = {row["file"] for row in csv.DictReader(file)}

    pool_context = nullcontext()  # no multiprocessing if WORKERS == 1
    if WORKERS > 1:
        print(f"Using {WORKERS} worker processes\n")
        pool_context = ProcessPoolExecutor(max_workers=WORKERS)

    try:
        with pool_context as pool:
            # run works just like the built-in map, either right here or spread across the workers,
            # so the steps don't need to know which
            run = map if pool is None else partial(pool.map, chunksize=CHUNK_SIZE)
            log_rows, counts = process_all(omit, run)
        report(log_rows, counts)
    finally:
        print(f"\nElapsed time: {time.perf_counter() - started:.0f} seconds")


if __name__ == "__main__":
    main()
