# Yes/No 



## Fine Tuning

The script `tune_yesno.py` can be used to fine tune BioBERT in any binary classified dataset if it's in the BioASQ format.

The use case presented, as in the paper, is fine-tuning on the BioASQ Yes/No dataset using as starting point the checkpoint from BioBERT fine-tuned on MNLI. (checkpoint `checkpoint_mnli.pt` on [drive](https://drive.google.com/drive/folders/1M45kWOrDTWjPZqJRE3Vo2RtrBMipCfV3?usp=sharing))

Example below:

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



## Predictions

To make predictions one can use the script `predict_yesno.py` with checkpoint  `checkpoint_yn.pt` on [drive](https://drive.google.com/drive/folders/1M45kWOrDTWjPZqJRE3Vo2RtrBMipCfV3?usp=sharing)

```python
python3 predict_yesno.py \
--model_name='dmis-lab/biobert-base-cased-v1.1' \
--checkpoint_input_path='../checkpoint/checkpoint_bio_yn.pt' \
--predictions_output_path='./predictions/pred_test.csv' \
--questions_path='../data/8B_golden/8B1_golden.json' \
--no-mid_layer
```

