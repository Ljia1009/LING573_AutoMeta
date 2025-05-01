# LING573_AutoMeta
For full original and training dataset, visit https://drive.google.com/drive/folders/14CXIUZWwPkoUQxVDcN8NLVOaYjwcPc-q?usp=drive_link

## Summarization
Command for running summarization from the repo root:
```bash
python src/run_summarization.py
```
Arguments:
```
--data_path:    Full path of the file used for testing.
                default=""
--data_option:  Option for the file used for testing;
                ignored when full path is provided;
                valid options are dev, test, or train
                default="dev"
--sample_size:  Number of samples to run summarization for;
                default to dataset length.
--key_option:   Option for the keys extracted from each review;
                Valid options are review, all.
                default="review"
--model:        Model used for summarization;
                valid options are bart, ...TBD
                default="bart"
--output_path:  Path to save the output.
                When unspecified, default to output/<model>_<key_option>_<sample_size>_out.txt
```

## Evaluation
Run evaluation using rougeL, bertscore, and factCC metrics from the repo root:
```bash
src/run_evaluation.sh
```

Run evaluation using disco metrics from the repo root:
```bash
src/run_evaluation_disco.sh
```

The evaluation results are save as csv files under `./evaluation` as `<metric>_<model>_<key_option>_out.txt.csv`
