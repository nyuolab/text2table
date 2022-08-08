import evaluate
from sklearn.metrics import classification_report

class Poseval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo()

    def _compute(
        self,
        predictions,
        references,
    ):
        report = classification_report(
            y_true=predictions,
            y_pred=references,
            output_dict=True,
            zero_division="warn",
        )

        return report