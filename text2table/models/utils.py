
# prediction (just a test)
def get_pred(pred):
    #log config:
    os.makedirs('pred_logs',exist_ok=True)
    date=datetime.datetime.now()
    n=date.strftime("pred_logs/%m_%d_%H:%M:%S_pred.log")
    pred_logger = setup_logger(name='pred_logger', log_file=n,formatter='%(levelname)s:%(message)s')
    pred_logger.info('\n---------Start of prediction epoch---------')

    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # Prepare the data for evaluation (as Text2Table task, we care about the special tokens)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
    for pred_str_row, label_str_row in zip(pred_str,label_str):
        #replace <pad> since annoying
        pred_logger.info(f"pred_str: {pred_str_row.replace('<pad>','')}")
        pred_logger.info(f"label_str: {label_str_row.replace('<pad>','')}")

    pred_logger.info('\n---------End of prediction epoch---------')

# predictions = trainer.predict(val_dataset)
# get_pred(predictions)