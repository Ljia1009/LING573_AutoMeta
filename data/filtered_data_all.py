import json

file_path = 'ORSUM_train.jsonl'
data_list = []

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        extracted_item = {'Review': [], 'Metareview': ''}
        key_to_exclude = ['title']
        for review in data['Review']:
            extracted_item['Review'].append(
                {k: v for k, v in review.items() if k not in key_to_exclude})
            if 'Metareview' in data:
                extracted_item['Metareview'] = data['Metareview']
        data_list.append(extracted_item)
