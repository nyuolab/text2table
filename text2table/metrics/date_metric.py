from datetime import date
from datetime import datetime
import evaluate
import datasets
from text2table.logging_utils.logging_script import setup_logger

def split_date(d):
    return datetime.strptime(d.split(' ')[0], "%Y-%m-%d").date()
# for cel-wise prediction/reference
class DateMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="wrong number of days",
            citation="NaN",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            )
        )

    def _compute(
        self,
        predictions,
        references,
    ):
        sum_days=0
        error_count=0
        for row_pred, row_ref in zip(predictions,references):
            try:
                pred = split_date(row_pred)
                ref = split_date(row_ref)
                sum_days+=abs((pred-ref).days)
            except ValueError:
                error_count+=1

        return {'days':sum_days/len(predictions),'wrong format count':error_count}
