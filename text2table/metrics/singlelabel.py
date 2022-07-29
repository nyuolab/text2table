# Reference: https://huggingface.co/docs/evaluate/a_wquick_tour#types-of-evaluations
import evaluate
import datasets

# Script to evaluate a single label
class SingleLabel(evaluate.Metric):
    # Initialize the metric
    def _info(self):
        return evaluate.MetricInfo(
            description="Single-Label Metric: F1, Precision, Recall",
            citation="NaN",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
            )
        )

    def _compute(self, predictions, references):
        # Load the evaluation metrics
        binary_metrics = evaluate.combine(["f1", "precision", "recall"])
        # Evaluation pipelines
        for ref, pred in zip(references, predictions):
            binary_metrics.add(references=ref, predictions=pred)    
        # Compute the metrics and return them
        return binary_metrics.compute()
