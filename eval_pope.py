import argparse
import json
import os


def eval_file(gt_file, gen_file):
    gt = [json.loads(q) for q in open(os.path.expanduser(gt_file), "r")]
    gen = [json.loads(q) for q in open(os.path.expanduser(gen_file), "r")]
    assert len(gt) == len(gen)
    tp = tn = fp = fn = 0
    pred_unknown = 0
    gt_unknown = 0
    yes_answers = 0
    for idx, line in enumerate(gt):
        gt_answer = line["label"].strip().lower()
        gen_answer = gen[idx]["text"].strip().lower()

        if "yes" in gen_answer:
            pred_label = "yes"
        elif "no" in gen_answer:
            pred_label = "no"
        else:
            pred_label = "unknown"

        if gt_answer == "yes":
            if pred_label == "yes":
                tp += 1; yes_answers += 1
            elif pred_label == "no":
                fn += 1
            else:
                fn += 1; pred_unknown += 1
        elif gt_answer == "no":
            if pred_label == "no":
                tn += 1
            elif pred_label == "yes":
                fp += 1; yes_answers += 1
            else:
                fp += 1; pred_unknown += 1
        else:
            gt_unknown += 1
            if pred_label == "yes":
                yes_answers += 1
            elif pred_label == "unknown":
                pred_unknown += 1

    total = len(gt) if len(gt) > 0 else 1
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    acc = (tp + tn) / total
    yes_prop = yes_answers / total
    pred_unknown_prop = pred_unknown / total
    gt_unknown_prop = gt_unknown / total
    return acc, precision, recall, f1, yes_prop, pred_unknown_prop, gt_unknown_prop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_files", type=str, required=True)
    parser.add_argument("--gen_files", type=str, required=True)
    args = parser.parse_args()
    acc, pre, rec, f1, yes, pred_unk, gt_unk = eval_file(args.gt_files, args.gen_files)
    print(f"Accuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nyes_pred: {yes:.4f}\npred_unknown: {pred_unk:.4f}\ngt_unknown: {gt_unk:.4f}")


if __name__ == "__main__":
    main()
