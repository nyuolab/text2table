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
        sum_days=0
        for row_pred, row_ref in zip(predictions,references):
            pred = split_date(row_pred)
            ref = split_date(row_ref)
            sum_days+=abs((pred-ref).days)
        return sum_days/len(predictions)




