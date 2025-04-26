from transformers import pipeline
import torch
def run_olmo_summarization(sample_size:int, data_list:list):
    if torch.cuda.is_available():
        # Get the ID of the first available GPU
        device_id = 0
    else:
        device_id = -1

    output = []
    gold_metareview = []

    for review in data_list[:sample_size]:
        gold_metareview.append(review['Metareview'])

    summarizer = pipeline("text-generation", model="allenai/OLMo-2-0325-32B",device_id = device_id)

    result = 'Below are multiple summaries of a paper\'s reviews. You need to summarize them. \n'
    for paper in data_list[:sample_size]:
        for review in paper['ReviewList']:
            summary = summarizer('summarize below review: '+review, max_new_tokens = 150,
                                min_length=60, do_sample=False)
            result += summary[0]['summary_text']+'\n'
        final = summarizer(result, max_new_tokens = 200,
                        min_length=90, do_sample=False)
        output.append(final[0]['summary_text'])
    return output, gold_metareview
