import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser("Data Preprocess")
parser.add_argument("--root_path", type=str, required=True)
parser.add_argument(
    "--label_order",
    type=str,
    default=None,
    help=(
        "Optional TSV containing the desired class order. The second column is "
        "used as the display name when present; otherwise the first column is used."
    ),
)
args = parser.parse_args()

root_path = args.root_path
os.chdir(root_path) 


def read_label_order(label_order):
    if label_order is None:
        label_order = "./label_order.tsv"
        if not os.path.exists(label_order):
            return None
    labels = []
    with open(label_order, newline="") as f:
        for line in f:
            row = line.rstrip("\n").split("\t")
            if not row or not row[0].strip():
                continue
            labels.append(row[1].strip() if len(row) > 1 else row[0].strip())
    return labels

train_meta = pd.read_csv("./train/train.tsv",sep="\t")
eval_meta = pd.read_csv("./eval/eval.tsv",sep="\t")

print("Total train files (in meta and we have):", len(pd.unique(train_meta["filename"])))
print("Total eval files (in meta and we have):", len(pd.unique(eval_meta["filename"])))
train_labels = pd.unique(train_meta["event_label"])
eval_labels = pd.unique(eval_meta["event_label"])

print("Total train labels:", len(set(train_labels)))
print("Total eval labels:", len(set(eval_labels)))
common_label_set = set(train_labels).intersection(set(eval_labels))
label_order = read_label_order(args.label_order)
if label_order is None:
    common_labels = sorted(common_label_set)
else:
    common_labels = [label for label in label_order if label in common_label_set]
    missing_labels = [label for label in label_order if label not in common_label_set]
    extra_labels = sorted(common_label_set.difference(common_labels))
    if missing_labels:
        print("Ordered labels missing from train/eval common set:", missing_labels)
    if extra_labels:
        print("Common labels missing from label order:", extra_labels)
        common_labels.extend(extra_labels)
print("Total common labels:", len(common_labels))
with open("./common_labels.txt", "w") as f:
    f.write("\n".join(common_labels))
# print("Disjoint train labels:", [x for x in set(train_labels) if x not in common_labels])
# print("Disjoint eval labels:", [x for x in set(eval_labels) if x not in common_labels])
train_rm_files = np.unique([filename for filename, label in zip(train_meta["filename"], train_meta["event_label"]) if label not in common_labels])
eval_rm_files = np.unique([filename for filename, label in zip(eval_meta["filename"], eval_meta["event_label"]) if label not in common_labels])
print("Train remove: ", len(train_rm_files), "Eval remove: ", len(eval_rm_files))

train_label_mask = np.array([True if x not in train_rm_files else False for x in train_meta["filename"] ])
eval_label_mask = np.array([True if x not in eval_rm_files else False for x in eval_meta["filename"] ])
train_meta_filt = train_meta[train_label_mask]
eval_meta_filt = eval_meta[eval_label_mask]

print("Total train files after filtering:", len(pd.unique(train_meta_filt["filename"])))
print("Total eval files after filtering:", len(pd.unique(eval_meta_filt["filename"])))

train_meta_filt.to_csv("./train/train_common.tsv", index=False, sep="\t")
eval_meta_filt.to_csv("./eval/eval_common.tsv", index=False, sep="\t")
