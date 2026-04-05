# vim: expandtab:ts=4:sw=4
import argparse
import os
import deep_sort_app
import motmetrics as mm
import pandas as pd

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="results")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value. Set to "
        "0.3 to reproduce results in the paper.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--filter", choices=["kf", "ekf", "ukf", "gas"], default="kf",
        help="Motion filter to use in the tracker (default: kf)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        deep_sort_app.run(
           sequence_dir, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, args.display, args.filter)

    mm.lap.default_solver = 'scipy'
    accs = []
    names = []
    for sequence in sequences:
        gt_file = os.path.join(args.mot_dir, sequence, "gt/gt.txt")
        res_file = os.path.join(args.output_dir, "%s.txt" % sequence)

        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        res = mm.io.loadtxt(res_file, fmt="mot15-2D")

        accs.append(mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5))
        names.append(sequence)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
