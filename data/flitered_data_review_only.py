import json

file_path = "/Users/xinyiliu/Desktop/Code_Test/LING573_AutoMeta/baseline/data/ORSUM_test.jsonl"
data_list = []

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        extracted_item = {'Review': [], 'Metareview': ''}
        if 'Metareview' in data:
            extracted_item['Metareview'] = data['Metareview']
            for review in data['Review']:
                if 'review' in review:
                    extracted_item['Review'].append(review['review'])
            data_list.append(extracted_item)
