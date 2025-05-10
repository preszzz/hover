import evaluate
import numpy as np

# --- Metrics Computation ---
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    metrics = accuracy_metric.compute(predictions=predictions, references=labels)
    metrics.update(precision_metric.compute(predictions=predictions, references=labels, average="binary"))
    metrics.update(recall_metric.compute(predictions=predictions, references=labels, average="binary"))
    metrics.update(f1_metric.compute(predictions=predictions, references=labels, average="binary"))
    return metrics