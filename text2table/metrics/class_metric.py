import pickle as pkl
import evaluate
from sklearn import metrics
import datasets

def dummify(data,classes):
    data=data.str.get_dummies(sep=' <CEL> ')
    # allign with index of all classes, solves the problem of unequal problems after dummification
    data=data.reindex(columns=classes).fillna(0)
    return data.to_numpy()

# for cel-wise prediction/reference
class DateMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=" ",
            citation=" ",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                    "classfile": datasets.Value("string", id="sequence"),
                }
            )
        )

    def _compute(
        self,
        predictions,
        references,
        classfile
    ):
        with open(classfile,'rb') as f:
            classes=pkl.load(f)
        dum_pred=dummify(predictions,classes)
        dum_ref=dummify(references,classes)

        f1=metrics.f1_score(dum_ref, dum_pred, average="micro")
        acc=metrics.accuracy_score(dum_ref, dum_pred)

        return {'micro_f1':f1,'acc':acc}