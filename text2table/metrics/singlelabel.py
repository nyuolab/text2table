# Reference: https://huggingface.co/docs/evaluate/a_wquick_tour#types-of-evaluations
import evaluate
import datasets
from sklearn import metrics

# Script to evaluate a single label
class SingleLabel(evaluate.Metric):
    # Initialize the metric
    def _info(self):
        return evaluate.MetricInfo(
            description="Single-Label Metric: F1, Precision, Recall",
            citation="NaN",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            )
        )

    def _compute(self, predictions, references):
        f1=metrics.f1_score(predictions, references, average="micro")
        acc=metrics.accuracy_score(predictions, references)
        return {'f1':f1,'acc':acc}
