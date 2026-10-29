"""
checks splits.csv for leakage between train, val and test. rerun whenever the split changes:

    pytest tests/test_split.py

run hash.py, dupe.py, standardize.py and split.py first
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "SeeNN"))
import split  # noqa: E402


def read_csv(path):
    assert path.exists(), f"{path.name} not found, run hash.py, dupe.py, standardize.py and split.py first"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crossing(rows, key):
    """returns {value: splits} for every value of key that appears in more than one split"""
    splits_of = defaultdict(set)
    for row in rows:
        splits_of[key(row)].add(row["split"])
    return {value: splits for value, splits in splits_of.items() if len(splits) > 1}


def test_no_group_crosses_splits():
    crossed = crossing(read_csv(split.SPLITS_CSV), lambda row: row["group"])
    assert not crossed, f"{len(crossed)} patient groups in more than one split, e.g. {list(crossed.items())[:5]}"


def test_no_patient_crosses_splits():
    crossed = crossing(read_csv(split.SPLITS_CSV), lambda row: row["patient"])
    assert not crossed, f"{len(crossed)} patients in more than one split, e.g. {list(crossed.items())[:5]}"


def test_no_duplicate_image_crosses_splits():
    """an image or its mirror must not appear in two splits, even under different patient IDs"""
    hashes = {row["file"]: row for row in read_csv(split.HASHES_CSV)}
    rows = [{"split": row["split"], "key": split.match_key(hashes[row["file"]])} for row in read_csv(split.SPLITS_CSV)]
    crossed = crossing(rows, lambda row: row["key"])
    assert not crossed, f"{len(crossed)} duplicate images in more than one split"


def test_official_test_set_unchanged():
    """every test image comes from the official test folder, and every official test image is kept"""
    rows = read_csv(split.SPLITS_CSV)
    official = {row["file"] for row in read_csv(split.HASHES_CSV) if row["split"] == "test"}
    in_test = {row["file"] for row in rows if row["split"] == "test"}
    assert in_test == official, f"{len(in_test ^ official)} images differ from the official test folder"
