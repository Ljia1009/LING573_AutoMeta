from transformers import pipeline
from flitered_data_review_only import data_list


output = []
gold_metareview = []

for review in data_list[]:
    gold_metareview.append(review['Metareview'])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

result = 'Below are multiple summaries of a paper\'s reviews. '
for paper in data_list[]:
    for review in paper['Review']:
        summary = summarizer(review, max_length=130,
                             min_length=30, do_sample=False)
        result += summary[0]['summary_text']+'\n'
    final = summarizer(result, max_length=130,
                       min_length=30, do_sample=False)
    output.append(final[0]['summary_text'])
