# LING573_AutoMeta
For full original and training dataset, visit https://drive.google.com/drive/folders/14CXIUZWwPkoUQxVDcN8NLVOaYjwcPc-q?usp=drive_link

Command for running summarization:
```bash
python src/run_summarization.py
```
Arguments:
```
--data_path:  default=""  Full path of the file used for testing.
--data_option:  default="dev"  Option for the file used for testing, ignored when full path is provided. Valid options are dev, test, or train
--key_option:  default="review" Option for the keys extracted from each review. Valid options are review, all.
--model:  default="bart"  Model uased for summarization. Valid options are bart, ...
--sample_size:  default=0  Number of samples to run summarization for, default to dataset length.
--output_path:  default="summarization_output.txt"  Path to save the output.
```
