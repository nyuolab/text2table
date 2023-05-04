# Experiments

## Requirements

Make sure to run `make requirements` before running any code. And you can also run
`make test_environment` to see you have met the requirments.

## Training and results

The code file we use to run experiments is `training.py`.
A example command to run the code is `python3 training.py --model bert --task PROC_ICD9-DIAG_ICD9 --top50 --tuning --freeze n`

Arguments:
```
--model (bert, roberta, or longformer): the language model trained and tested. 
--task (main, main+aux): main tasks include "PROC_ICD9" or "DIAG_ICD9" and auxiliary tasks include "PROC_ICD9", "DIAG_ICD9", "CPT_CD", or "DRG_CODE".
--top50: argument for whether you use MIMIC-III-50
--tuning: argument for whether you do threshold tuning
--freeze (n, or f): "n" means you do not freeze the language model base. "f" means you freeze the language model base.
```