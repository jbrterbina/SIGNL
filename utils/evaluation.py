import numpy as np
from sklearn.metrics import confusion_matrix


def compute_eer(scores, labels):
    if isinstance(scores, list) is False:
        scores = list(scores)
    if isinstance(labels, list) is False:
        labels = list(labels)

    target_scores = []
    nontarget_scores = []

    for item in zip(scores, labels):
        if item[1] == 1:
            target_scores.append(item[0])
        else:
            nontarget_scores.append(item[0])

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    target_position = 0
    for i in range(target_size - 1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    return eer, th


# def compute_metrics(labels, scores):
def compute_eer_ori(labels, scores):
    try:
        eer, th = compute_eer(scores, labels)
        preds = np.where(scores >= th, 1, 0)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Calculate Precision, Recall, and F1 Score
        precision = tp / (tp + fp + 1e-4)
        recall = tp / (tp + fn + 1e-4)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-4)

        print("True Negatives (tn):", tn)
        print("False Positives (fp):", fp)
        print("False Negatives (fn):", fn)
        print("True Positives (tp):", tp)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1_score)
        print("Equal Error Rate (EER): {:.2f}%\n".format(eer * 100))

        return {
            "EER": eer,
            "ACC": accuracy,
            "F1": f1_score,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "TP": tp,
        }, th

    except Exception as e:
        print("An error occurred:", e)
        eer = 0  # Set EER to 0 in case of an error
        thresholds = 0

        return eer, thresholds
