"""
hashes all images in the dataset
for every image in the default train/test subfolders of the OCT dataset, this script 
computes + saves a fingerprint of the image's pixels and a fingerprint of the same 
image flipped left to right, then writes everything to image_hashes.csv
"""
import csv
import hashlib
from pathlib import Path
from PIL import Image

# settings, defaulting to local OCT dataset folder and output CSV file in the same folder as this script
DATA_FOLDER = Path(__file__).parent / "OCT"
# can use global path instead using r string
# DATA_FOLDER = Path(r"C:\whatever\your\path\is\to\OCT")
OUTPUT_CSV = Path(__file__).parent / "image_hashes.csv"

SPLITS = ["train", "test"]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def fingerprint(image):
    """calculates a sha256 hash returned in hex"""
    header = f"{image.mode} {image.width}x{image.height} ".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def extract_id(file_name):
    """returns the second part of a file name like CNV-1016042-1.jpeg, or an empty string if the name doesn't fit the pattern"""
    parts = Path(file_name).stem.split("-")
    if len(parts) == 3:
        return parts[1]
    return ""


def find_images():
    """returns [split, class, path] for every image in the dataset"""
    found = []
    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATA_FOLDER / split / class_name
            if not folder.is_dir():
                print(f"Err: skipping absent: {folder}")
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in (".jpeg", ".jpg"):
                    found.append((split, class_name, path))
    return found


def main():
    # error out if the data folder doesn't exist, or if no images are found
    if not DATA_FOLDER.is_dir():
        print(f"DATA_FOLDER does not exist: {DATA_FOLDER}")
        print("Edit DATA_FOLDER at the top of this file and run it again.")
        return

    image_files = find_images()
    if not image_files:
        print("No images found")
        return
    print(f"Fingerprinting {len(image_files):,} images")

    rows = []
    unreadable = 0
    for number, (split, class_name, path) in enumerate(image_files, start=1):
        row = {
            "file": path.relative_to(DATA_FOLDER).as_posix(),  #folder and filename relative to dataset
            "split": split,
            "class": class_name,
            "patient": extract_id(path.name),
            "readable": "yes",
            "pixel_hash": "",
            "mirror_hash": "",
            "error": "",
        }
        try:
            with Image.open(path) as image:
                image.load()  # load image
                row["pixel_hash"] = fingerprint(image)
                flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                row["mirror_hash"] = fingerprint(flipped)
        except Exception as error: # if unreadable, note as such and append to counter
            row["readable"] = "no"
            row["error"] = str(error)
            unreadable += 1
        rows.append(row)
        if number % 5000 == 0:
            print(f"{number:,} of {len(image_files):,} done")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for row in rows if row["split"] == "train")
    n_test = sum(1 for row in rows if row["split"] == "test")
    print()
    print("Hashing complete.")
    print(f"Images found: {len(rows):,}  (train {n_train:,}, test {n_test:,})")
    print(f"Unreadable: {unreadable:,}")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()