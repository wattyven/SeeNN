"""
reads our list of hashes, finds dupes, writes list of images to leave out to omit_list.csv
"""
import csv
from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parents[2] / "data" / "OCT"
HASHES_CSV = DATA_FOLDER / "image_hashes.csv"
OMIT_CSV = DATA_FOLDER / "omit_list.csv"


def match_key(row):
    """matches images that are the either identical or flipped left to right."""
    return min(row["pixel_hash"], row["mirror_hash"]) # this works because for a pair of images, a source and its mirror,
    # the same image between the two will always have the smaller hex value,
    # so if images are identical or mirror images, between their two keys they'll return
    # the same minimum value


def omit_entry(row, reason, group):
    """generates row for the csv file"""
    others = [other["file"] for other in group if other["file"] != row["file"]]
    return {
        "file": row["file"],
        "split": row["split"],
        "class": row["class"],
        "patient": row["patient"],
        "reason": reason,
        "same_image_as": " | ".join(others),
    }


def main():
    if not HASHES_CSV.exists():
        print(f"{HASHES_CSV.name} not found, please compute hashes by running hash.py first")
        return
    with open(HASHES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    to_omit = []

    # group images by pixel/mirror hash
    # first copy refers to the first image in a group, the rest are duplicates
    groups = {}
    for row in rows:
        if row["readable"] != "yes":
            to_omit.append(omit_entry(row, "could not be opened", []))
            continue
        key = match_key(row)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    dupe_groups = [group for group in groups.values() if len(group) > 1] # heh it rhymes

    # counters
    grouped_imgs = 0
    across_patients = 0
    across_splits = 0
    conflicting = 0
    with_mirror = 0
    extra_test_copies = 0

    for group in dupe_groups:
        train_copies = [row for row in group if row["split"] == "train"]
        test_copies = [row for row in group if row["split"] == "test"]
        classes = set(row["class"] for row in group)
        patients = set(row["patient"] for row in group)

        grouped_imgs += len(group)
        if len(patients) > 1:
            across_patients += 1
        if train_copies and test_copies:
            across_splits += 1
        if len(test_copies) > 1:
            extra_test_copies += len(test_copies) - 1
        first = group[0]
        if any(row["pixel_hash"] != first["pixel_hash"] for row in group):
            with_mirror += 1  # has a mirror image of the source

        if len(classes) > 1:
            conflicting += 1
            for row in train_copies:
                to_omit.append(omit_entry(row, "conflicting labels", group))
        elif test_copies:
            for row in train_copies:
                to_omit.append(omit_entry(row, "copy of a test image", group))
        else:
            for row in train_copies[1:]:
                to_omit.append(omit_entry(row, "extra copy within training", group))

    with open(OMIT_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["file", "split", "class", "patient", "reason", "same_image_as"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(to_omit)

    reasons = ["could not be opened", "conflicting labels", "copy of test image", "extra copy within training"]

    print(f"Images checked: {len(rows):,}")
    print(f"Dupe groups: {len(dupe_groups):,} with {grouped_imgs:,} images")
    print(f"Dupes across patients (these patients must be put into the same split): {across_patients:,}")
    print(f"Dupes across train and test: {across_splits:,}")
    print(f"Dupes with conflicting labels: {conflicting:,}")
    print(f"DUpes containing a mirror: {with_mirror:,}")
    print(f"Extra copies in test: {extra_test_copies:,}")
    print()
    print(f"Images to omit: {len(to_omit):,}")
    for reason in reasons:
        count = sum(1 for entry in to_omit if entry["reason"] == reason)
        print(f"  {reason + ':':<28}{count:,}")
    print(f"Written to: {OMIT_CSV}")


if __name__ == "__main__":
    main()
