

## Tuning



```python
python3 tune_list.py \
--model_name='dmis-lab/biobert-base-cased-v1.1' \
--checkpoint_input_path='../checkpoint/checkpoint_mnli_squad.pt' \
--checkpoint_output_path='../checkpoint' \
--bioasq_path='../data/bioasq_train_9b/training9b.json' \
--learning_rate=5e-5 \
--batch_size=16 \
--epochs=3
```



## Predict

```python
python3 predict_list.py \
--model_name='dmis-lab/biobert-base-cased-v1.1' \
--checkpoint_input_path='../checkpoint/checkpoint_list.pt' \
--predictions_output_path='./predictions/pred.csv' \
--questions_path='../data/8B_golden/8B1_golden.json' \
--k_candidates=5 \
--k_elected=2 \
--voting='stv'
```

