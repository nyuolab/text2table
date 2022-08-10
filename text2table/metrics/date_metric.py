from datetime import date
from datetime import datetime

def split_date(d):
    return datetime.strptime(d.split(' ')[0], "%Y-%m-%d").date()
# for cel-wise prediction/reference
class DateMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo()

    def _compute(
        self,
        predictions,
        references,
    ):
        pred = split_date(predictions)
        ref = split_date(references)
        return abs((pred-ref).days)




