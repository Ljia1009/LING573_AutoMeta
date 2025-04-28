import json

# 打开 jsonl 文件
with open('data/ORSUM_test.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    print(len(data))