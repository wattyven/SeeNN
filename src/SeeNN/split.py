"""
splits the cleaned images into train, val, and test, then writes splits.csv

the official test folder is kept unchanged, whereas the images in the ORIGINAL training split are 
further split into train and val by patient, so every image of a patient lands in the same set. 
this is done by recording the new split label in a csv rather than modifying the folder structure directly.
patients that share an identical image anywhere in the raw training data are merged into one group first 
since two IDs with the same scan are probably still the same person, just mislabelled 

run hash.py, dupe.py and standardize.py first
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

DATA_FOLDER = Path(__file__).resolve().parents[2] / "data" / "OCT"
HASHES_CSV = DATA_FOLDER / "image_hashes.csv"
LOG_CSV = DATA_FOLDER / "standardize_log.csv"
SPLITS_CSV = DATA_FOLDER / "splits.csv"

SEED = 7615
VAL_FRACTION = 0.2 # proportion of training data to assign to validation
# best to use a number that divides evenly into 1/VAL_FRACTION, to balance folds
# otherwise we round
# ex 0.2 = 5 folds, 0.1 = 10 folds, but 0.15 leads to 7 folds
SPLITS = ["train", "val", "test"]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def match_key(row):
    """same as in dupe.py"""
    return min(row["pixel_hash"], row["mirror_hash"])


def patient_groups(rows):
    """returns {patient: group}, where patients that share any duplicate image get the same group. groups are named by their smallest patient ID"""
    parent = {}

    def find(patient): # find the root, compressing the path along the way
        parent.setdefault(patient, patient)
        while parent[patient] != patient: # if the patient is not its own parent, follow the chain to find the root
            parent[patient] = parent[parent[patient]]  # shortcut the path as we go
            patient = parent[patient]
        return patient

    def union(a, b): # attach BIG root to smol root
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b) # always point the larger ID to the smaller one

    patients_by_key = defaultdict(set) # maps each image hash to associated patient IDs
    for row in rows:
        find(row["patient"])
        if row["readable"] == "yes": # only consider readable images for grouping
            patients_by_key[match_key(row)].add(row["patient"]) # group patients by the unique image hash
    for patients in patients_by_key.values():
        first, *rest = sorted(patients) # sort IDs so the same root is chosen, ignoring order
        for other in rest:
            union(first, other)
    return {patient: find(patient) for patient in parent} # returns a mapping of each patient to the root of their group, thus merging patients that share ANY duplicated images


def assign_val(train_rows):
    """returns patient groups in validation set"""
    folds = StratifiedGroupKFold(n_splits=round(1 / VAL_FRACTION), shuffle=True, random_state=SEED) # stratified k fold splitter
    labels = [row["class"] for row in train_rows]
    groups = [row["group"] for row in train_rows]
    _, val_index = next(folds.split(train_rows, labels, groups)) # use the first fold as the validation set
    return {groups[i] for i in val_index} # returns the set of unique groups that are assigned to the validation set


def main():
    for path in [HASHES_CSV, LOG_CSV]:
        if not path.exists():
            print(f"{path.name} not found, please first run hash.py, dupe.py and standardize.py IN THAT ORDER")
            return
    with open(HASHES_CSV, newline="", encoding="utf-8") as f:
        hashes = list(csv.DictReader(f))
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        outputs = {row["file"]: row["output"] for row in csv.DictReader(f)}

    group_of = patient_groups(hashes)
    # only images that made it through standardize.py are split, everything else was omitted
    rows = [{"file": row["file"], "output": outputs[row["file"]], "split": row["split"],
             "class": row["class"], "patient": row["patient"], "group": group_of[row["patient"]]}
            for row in hashes if row["file"] in outputs]

    train_rows = [row for row in rows if row["split"] == "train"] # only consider the training set for splitting into train/val
    val_groups = assign_val(train_rows)
    for row in train_rows:
        if row["group"] in val_groups:
            row["split"] = "val"

    with open(SPLITS_CSV, "w", newline="", encoding="utf-8") as f: # write the final splits to csv
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows) 

    merged = Counter(group_of.values()) # count how many patients are in each group
    print(f"Patients: {len(group_of):,} in {len(merged):,} groups "
          f"({sum(1 for n in merged.values() if n > 1)} groups merge more than one patient)")
    print()
    print(f"{'':<7}{'images':>8}{'patients':>10}{'groups':>8}" + "".join(f"{c:>8}" for c in CLASSES)) # print header for the summary table
    for split in SPLITS:
        part = [row for row in rows if row["split"] == split]
        classes = Counter(row["class"] for row in part)
        print(f"{split:<7}{len(part):>8,}{len(set(r['patient'] for r in part)):>10,}"
              f"{len(set(r['group'] for r in part)):>8,}" + "".join(f"{classes[c]:>8,}" for c in CLASSES)) # print summary for each split
    print(f"\nOutput: {SPLITS_CSV}")


if __name__ == "__main__":
    main()