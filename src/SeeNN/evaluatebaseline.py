"""
Evaluates all models on the validation split using saved predictions.

Uses macro F1 as the primary metric (equal weight for all four classes, regardless of frequency). 
Secondary metrics are accuracy, per-class F1, urgent referral sensitivity and specificity (CNV/DME), 
and DME recall by scan size (512×512 vs other sizes, the shortcut identified at M3). Each prediction is one B-scan.

Reads results/<model>/predictions_val_seed<SEED>_<SETTINGS>.csv (one per seed) for each model in MODELS, and
results/<model>/sweep.csv where a model has one (the logistic regression's train and val loss and val macro F1 for
each C tried, and which C was chosen).
Writes results/runs.csv (per-run metrics), results/results.csv (per-model mean, with the sd across seeds when there
are several), and results/figures/ (a confusion matrix and error examples per model, and a tuning figure from each
sweep.csv). The most-common-class baseline requires no predictions file to compute and the test split is never used.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from split import CLASSES, DATA_FOLDER, SEED, SPLITS_CSV
from standardize import LOG_CSV, OUTPUT_FOLDER

ROOT = DATA_FOLDER.parents[1]  # repo root
RESULTS_FOLDER = ROOT / "results"
FIGURES_FOLDER = RESULTS_FOLDER / "figures"
MODELS = {"logitbaseline": "HOG + logistic regression"}  
# when small cnn ready, add the following entry to the models dict
# "cnnbaseline": shallow CNN
# this eval will work so long as the cnnbaseline model 
# writes outputs to results/cnnbaseline/predictions_val_SEED_SETTINGS.csv
URGENT = ["CNV", "DME"]
ERROR_KINDS, ERROR_EXAMPLES = 6, 4  # most common kinds of error, and random examples of each, in the errors figure
# how the table is printed on screen: (heading, {column in results.csv: label in plain words}), in order
PRINTED = [(None, {"macro_f1": "macro F1 (primary metric)", "accuracy": "accuracy"}),
           ("F1 by class", {f"f1_{c}": c for c in CLASSES}),
           ("urgent referral (CNV or DME)", {"urgent_sensitivity": "sensitivity: urgent scans flagged urgent", 
                                             "urgent_specificity": "specificity: other scans not flagged"}),
           ("shortcut check: share of DME scans called DME", {"dme_recall_512x512": "512x512 scans", "dme_recall_other_sizes": "other sizes"})]

INK, MUTED, GRID, LINE = "#2b2b2b", "#5f6b7a", "#e3e5e8", "#2a78d6"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.constrained_layout.use": True, "savefig.dpi": 200})


def read_split(name):
    """the train or val rows of splits.csv, in file order, with each scan's original size"""
    if name not in ("train", "val"):
        raise ValueError("only train and val are used before the final evaluation")
    splits = pd.read_csv(SPLITS_CSV, dtype=str)
    log = pd.read_csv(LOG_CSV, usecols=["file", "width", "height"])
    rows = splits[splits["split"] == name].merge(log, on="file", how="left")
    rows["size"] = rows["width"].astype(str) + "x" + rows["height"].astype(str)
    return rows.drop(columns=["width", "height"]).reset_index(drop=True)


def confusion(true, pred):
    """counts of each true class (rows) predicted as each class (columns), from class indices"""
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (true, pred), 1)
    return cm


def f1_per_class(cm):
    """F1 = 2TP / (2TP + FP + FN) of each class, 0 for a class that is never predicted"""
    tp = np.diag(cm)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)


def macro_f1(cm):
    return f1_per_class(cm).mean()


def scores(true, pred, sizes):
    """every metric in the results table, from arrays of true and predicted class names and original scan sizes"""
    cm = confusion([CLASSES.index(c) for c in true], [CLASSES.index(c) for c in pred])
    urgent, called = np.isin(true, URGENT), np.isin(pred, URGENT)
    dme = true == "DME"
    return {"macro_f1": macro_f1(cm), "accuracy": (true == pred).mean(),
            **{f"f1_{c}": f for c, f in zip(CLASSES, f1_per_class(cm))},
            "urgent_sensitivity": called[urgent].mean(), "urgent_specificity": (~called[~urgent]).mean(),
            "dme_recall_512x512": (pred[dme & (sizes == "512x512")] == "DME").mean(),
            "dme_recall_other_sizes": (pred[dme & (sizes != "512x512")] == "DME").mean()}


def summary_table(runs):
    """one row per model with each metric's mean and the standard deviation across seeds when there are several"""
    rows = []
    for model, group in runs.groupby("model", sort=False):
        row = {"model": model, "seeds": len(group) if group["seeds"].iloc[0] else "none, no training"}
        for metric in [c for c in runs.columns if c not in ("model", "seeds", "file")]:
            mean, sd = group[metric].mean(), group[metric].std()
            row[metric] = f"{mean:.3f} ± {sd:.3f}" if len(group) > 1 else f"{mean:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(table, scans):
    """results table with one column per model"""
    models = table["model"].tolist()
    table = table.set_index("model").astype(str).replace("none, no training", "no training")
    rows = [("runs (one per seed)", "seeds")]
    for heading, metrics in PRINTED:
        if heading:
            rows += [("", None), (heading, None)]
        rows += [(("  " if heading else "") + label, column) for column, label in metrics.items()]
    label_width = max(len(label) for label, _ in rows)
    widths = [max(len(model), *map(len, table.loc[model])) for model in models]
    print(f"validation set ONLY: {scans:,} scans\n")
    print(" " * label_width + "".join(f"   {model:>{w}}" for model, w in zip(models, widths)))
    for label, column in rows:
        cells = "".join(f"   {table.loc[model, column]:>{w}}" for model, w in zip(models, widths)) if column else ""
        print(f"{label:<{label_width}}{cells}".rstrip())
    print("\nscores run from 0 to 1 where higher is better. models with several runs will show the mean ± sd across seeds.")


# figures (saved to results/figures/)

def confusion_figure(true, pred, name, path):
    cm = confusion([CLASSES.index(c) for c in true], [CLASSES.index(c) for c in pred])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, values, text, title in [(axes[0], cm, "{:,.0f}", "(a) val scans"),
                                    (axes[1], 100 * cm / cm.sum(axis=1, keepdims=True), "{:.1f}%",
                                     "(b) share of each true class (%)")]:
        ax.imshow(values, cmap="Blues", vmin=0)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, text.format(values[i, j]), ha="center", va="center",
                        color="white" if values[i, j] > 0.55 * values.max() else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set(xlabel="predicted class", ylabel="true class", title=title)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(f"{name}: val confusion matrix, macro F1 {macro_f1(cm):.3f}")
    fig.savefig(path)
    plt.close(fig)


def errors_figure(predictions, name, path):
    """random val scans as standardized from the most common kinds of error"""
    wrong = predictions[predictions["class"] != predictions["pred"]]
    kinds = wrong.groupby(["class", "pred"]).size().sort_values(ascending=False).head(ERROR_KINDS)
    fig, axes = plt.subplots(len(kinds), ERROR_EXAMPLES, figsize=(7.2, 1.95 * len(kinds)), squeeze=False)
    for row_axes, ((true, pred), count) in zip(axes, kinds.items()):
        examples = wrong[(wrong["class"] == true) & (wrong["pred"] == pred)]
        examples = examples.sample(min(ERROR_EXAMPLES, len(examples)), random_state=SEED)
        for ax, (_, row) in zip(row_axes, examples.iterrows()):
            with Image.open(OUTPUT_FOLDER / row["output"]) as image:
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"p({pred}) {row[f'p_{pred}']:.2f}, {row['size']}", fontsize=7)
        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(f"true {true}\npredicted {pred}\n({count:,} scans)")
    fig.suptitle(f"{name}: random val scans from the {len(kinds)} most common errors\n"
                 "each titled with the probability given to the wrong class and the original scan size")
    fig.savefig(path)
    plt.close(fig)


def tuning_figure(sweep, chosen, path):
    """the logistic regression's train and val loss, and val macro F1, against C. drawn with one quantity per axis, so two panels"""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for column, label, color in [("train_weighted_log_loss", "train", MUTED), ("val_weighted_log_loss", "val", LINE)]:
        ax_a.plot(sweep["C"], sweep[column], color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax_a.set(ylabel="class-weighted log loss", title="(a) loss: val rises once C lets the model overfit")
    ax_a.legend(frameon=False)
    ax_b.plot(sweep["C"], sweep["val_macro_f1"], color=LINE, linewidth=2, marker="o", markersize=4)
    ax_b.set(ylabel="val macro F1", title="(b) val macro F1, the primary metric")
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xticks(sweep["C"], [f"{c:g}" for c in sweep["C"]])
        ax.minorticks_off()
        ax.set_xlabel("C (inverse regularization strength, log scale)")
        ax.axvline(chosen, color=INK, linestyle="--", linewidth=1)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"HOG + logistic regression: tuning C on val (dashed: chosen C = {chosen:g})")
    fig.savefig(path)
    plt.close(fig)


def main():
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    val = read_split("val")
    true, sizes = val["class"].to_numpy(), val["size"].to_numpy()
    most_common = read_split("train")["class"].mode()[0]
    runs = [{"model": f"most common class ({most_common})", "seeds": 0, "file": "",
             **scores(true, np.full(len(val), most_common), sizes)}]

    for folder, name in MODELS.items():
        files = sorted((RESULTS_FOLDER / folder).glob("predictions_val*.csv"))
        for k, path in enumerate(files):
            predictions = pd.read_csv(path, dtype={"size": str})
            if predictions["file"].tolist() != val["file"].tolist():
                raise ValueError(f"{path} does not hold the val split in splits.csv order")
            predictions["output"] = val["output"]
            runs.append({"model": name, "seeds": len(files), "file": path.relative_to(ROOT).as_posix(),
                         **scores(true, predictions["pred"].to_numpy(), sizes)})
            if k == 0:  # figures from the first seed
                confusion_figure(true, predictions["pred"].to_numpy(), name, FIGURES_FOLDER / f"confusion_{folder}.png")
                errors_figure(predictions, name, FIGURES_FOLDER / f"errors_{folder}.png")
        sweep_file = RESULTS_FOLDER / folder / "sweep.csv"
        if sweep_file.exists():
            sweep = pd.read_csv(sweep_file)
            chosen = sweep.loc[sweep["chosen"], "C"].iloc[0]
            tuning_figure(sweep, chosen, FIGURES_FOLDER / f"tuning_{folder}.png")

    runs = pd.DataFrame(runs)
    runs.round(4).to_csv(RESULTS_FOLDER / "runs.csv", index=False)
    table = summary_table(runs)
    table.to_csv(RESULTS_FOLDER / "results.csv", index=False, encoding="utf-8-sig")  # utf-8-sig so Excel properly displays ±
    print_table(table, len(val))
    print(f"\nwrote results/runs.csv, results/results.csv and {len(list(FIGURES_FOLDER.glob('*.png')))} figures in "
          "results/figures/")


if __name__ == "__main__":
    main()