```python
python3 tune_yesno.py \
--model_name='dmis-lab/biobert-base-cased-v1.1' \
--checkpoint_input_path='../checkpoint/checkpoint_mnli.pt' \
--checkpoint_output_path='../checkpoint' \
--bioasq_path='../data/bioasq_train_9b/training9b.json' \
--learning_rate=5e-5 \
--batch_size=16 \
--epochs=3 \
--no-mid_layer
```



```python
python3 predict_yesno.py \
--model_name='dmis-lab/biobert-base-cased-v1.1' \
--checkpoint_input_path='../checkpoint/checkpoint_bio_yn_256.pt' \
--predictions_output_path='./predictions/pred_test.csv' \
--questions_path='../data/8B_golden/8B1_golden.json' \
--no-mid_layer
```

