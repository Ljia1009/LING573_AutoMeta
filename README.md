# LING573_AutoMeta
For full original and training dataset, visit https://drive.google.com/drive/folders/14CXIUZWwPkoUQxVDcN8NLVOaYjwcPc-q?usp=drive_link

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
                default="summarization_output.txt"
```
