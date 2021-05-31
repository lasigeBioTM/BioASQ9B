# Fine Tuning BioBERT











## Run Script

```python
python3 tune_mnli_squad.py \
--model_name='dmis-lab/biobert-base-cased-v1.1' \
--checkpoint_output_path='../checkpoint' \
--mnli_path='../data/multinli_1.0_train.json' \
--squad_path='../data/squad_train-v2.0.json' \
--learning_rate=5e-5 \
--batch_size=16 \
--epochs=3
```











