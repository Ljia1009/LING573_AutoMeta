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
        for review in data['Review']:
            if 'Metareview' in data:
                extracted_item['Metareview'] = data['Metareview']
            if 'review' in review:
                extracted_item['Review'].append(review['review'])
        data_list.append(extracted_item)
