def compute_metrics(EvalPrediction, tokenizer):
    predictions = EvalPrediction.predictions
    label_ids = EvalPrediction.label_ids
    inputs = EvalPrediction.inputs

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=False)
    
