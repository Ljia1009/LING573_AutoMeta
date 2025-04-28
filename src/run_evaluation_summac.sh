#!/bin/sh
DATA_PATH="/Users/chenxinliu/LING573_AutoMeta/data/ORSUM_test.jsonl"
KEY_OPTION="all"
OUTPUT_PATH="/Users/chenxinliu/LING573_AutoMeta/output/bart_all_out.json"
EVAL_PATH="/Users/chenxinliu/LING573_AutoMeta/evaluation/bart_all_out_summac.csv"

python run_evaluation_summac.py --data_path $DATA_PATH --key_option $KEY_OPTION --output_path $OUTPUT_PATH --evaluation_result_path $EVAL_PATH