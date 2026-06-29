# vim: expandtab:ts=4:sw=4
"""Grid search over ``max_cosine_distance`` using the Kalman filter (``kf``).

For each value in {0.1, 0.15, 0.2, 0.25, 0.3, 0.4} we run the DeepSORT
pipeline over every sequence in ``--mot_dir`` and compute MOTChallenge
metrics. The summary per value (and the per-sequence breakdown) is written
to ``--output_dir`` so the best threshold can be picked from MOTA / IDF1.

Example
-------
python tests/gridsearch_max_cosine.py \
    --mot_dir /path/to/MOT16/train \
    --output_dir results/gridsearch_max_cosine
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import motmetrics as mm
import pandas as pd

# Make the repo root importable when this script is executed from anywhere.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_sort_app  # noqa: E402  (import after sys.path tweak)


GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
FILTER = "kf"


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please enter a valid True/False choice")
    return input_string == "True"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search max_cosine_distance with Kalman filter"
    )
    parser.add_argument(
        "--mot_dir",
        help="Path to MOTChallenge directory (train or test).",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where per-value tracking outputs and the summary "
        "CSV will be written.",
        default="results/gridsearch_max_cosine",
    )
    parser.add_argument(
        "--min_confidence", type=float, default=0.3,
        help="Detection confidence threshold (forwarded to deep_sort_app).",
    )
    parser.add_argument(
        "--min_detection_height", type=int, default=0,
        help="Minimum detection height (forwarded to deep_sort_app).",
    )
    parser.add_argument(
        "--nms_max_overlap", type=float, default=1.0,
        help="NMS max overlap (forwarded to deep_sort_app).",
    )
    parser.add_argument(
        "--nn_budget", type=int, default=100,
        help="Appearance gallery budget (forwarded to deep_sort_app).",
    )
    parser.add_argument(
        "--display", type=bool_string, default=False,
        help="Whether to show intermediate visualization (default False).",
    )
    parser.add_argument(
        "--primary_metric", default="mota",
        choices=["mota", "idf1", "motp"],
        help="Metric used to choose the winning max_cosine_distance.",
    )
    return parser.parse_args()


def evaluate_one(mot_dir, output_dir, max_cosine_distance, args):
    """Run the tracker for every sequence in ``mot_dir`` and return metrics.

    Returns
    -------
    (overall_summary_row : pandas.Series, per_seq_summary : pandas.DataFrame)
    """
    os.makedirs(output_dir, exist_ok=True)
    sequences = sorted(
        s for s in os.listdir(mot_dir)
        if os.path.isdir(os.path.join(mot_dir, s))
    )

    for sequence in sequences:
        sequence_dir = os.path.join(mot_dir, sequence)
        output_file = os.path.join(output_dir, "%s.txt" % sequence)
        print("[max_cosine_distance=%.3f] running sequence %s"
              % (max_cosine_distance, sequence))
        deep_sort_app.run(
            sequence_dir, output_file,
            args.min_confidence, args.nms_max_overlap,
            args.min_detection_height, max_cosine_distance,
            args.nn_budget, args.display, FILTER,
        )

    mm.lap.default_solver = "scipy"
    accs, names = [], []
    for sequence in sequences:
        gt_file = os.path.join(mot_dir, sequence, "gt/gt.txt")
        res_file = os.path.join(output_dir, "%s.txt" % sequence)
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        res = mm.io.loadtxt(res_file, fmt="mot15-2D")
        accs.append(mm.utils.compare_to_groundtruth(gt, res, "iou", distth=0.5))
        names.append(sequence)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=True,
    )
    print(mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))
    return summary.loc["OVERALL"].copy(), summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    per_value_summaries = {}
    for mcd in GRID:
        tag = ("mcd_%.3f" % mcd).rstrip("0").rstrip(".")
        run_out_dir = os.path.join(args.output_dir, tag)
        t0 = time.time()
        overall, full_summary = evaluate_one(
            args.mot_dir, run_out_dir, mcd, args,
        )
        elapsed = time.time() - t0

        per_value_summaries[mcd] = full_summary
        full_summary.to_csv(os.path.join(run_out_dir, "per_sequence.csv"))

        row = overall.to_dict()
        row["max_cosine_distance"] = mcd
        row["elapsed_sec"] = elapsed
        rows.append(row)
        print("[max_cosine_distance=%.3f] OVERALL %s=%s (%.1fs)"
              % (mcd, args.primary_metric,
                 overall.get(args.primary_metric, "n/a"), elapsed))

    results = pd.DataFrame(rows).set_index("max_cosine_distance")
    # Put the chosen metric first for readability.
    cols = [args.primary_metric] + [
        c for c in results.columns if c != args.primary_metric
    ]
    results = results[cols].sort_values(
        args.primary_metric,
        ascending=(args.primary_metric == "motp"),  # motp: lower is better
    )
    summary_path = os.path.join(args.output_dir, "gridsearch_summary.csv")
    results.to_csv(summary_path)

    best_value = results.index[0]
    best_score = results.iloc[0][args.primary_metric]
    print("\n=== Grid search complete ===")
    print(results.to_string())
    print("\nBest max_cosine_distance by %s: %.3f (%s=%s)"
          % (args.primary_metric, best_value,
             args.primary_metric, best_score))
    print("Summary written to %s" % summary_path)


if __name__ == "__main__":
    main()
