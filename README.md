# BioASQ 9B LASIGE_ULISBOA

All code developed in  Python 3.6.0

Required packages in `requirements.txt`



## Required Data

All important data and model checkpoints can be saved can be accessed through:

https://drive.google.com/drive/folders/1M45kWOrDTWjPZqJRE3Vo2RtrBMipCfV3?usp=sharing

For the examples provided to work save all contents of drive folder `chekpoints` into repository folder `checkpoint`, and all data from drive folder `datasets` into repository folder `data`



### Format BioASQ

This script receives a file in BioASQ format, containing questions as well as paths for predictions (.csv format) for list, factoid and yes/no questions and returns the predictions file in BioASQ format. This file can be used with golden files for official evaluation in the [official repo.](https://github.com/BioASQ/Evaluation-Measures/tree/master/flat/BioASQEvaluation)

```python
python3 format_bioasq.py \
--questions_path='./data/8B_golden/8B1_golden.json' \
--list_predictions_path='./list/predictions/pred.csv' \
--factoid_predictions_path='./factoid/predictions/pred.csv' \
--yesno_predictions_path='./yesno/predictions/pred_test.csv' \
--output_path='./predictions/pred_bioasq.json'
```

