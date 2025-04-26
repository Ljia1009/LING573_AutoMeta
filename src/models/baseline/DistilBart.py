from transformers import pipeline
import torch
def run_distilbart_summarization(sample_size:int, data_list:list):
    """
    Run the summarization process with BART.
    """
    # Load the data

    # Initialize the output list and gold metareview list
    output = []
    gold_metareview = []

    # Extract the gold metareview from the first specified reviews
    for review in data_list[:sample_size]:
        gold_metareview.append(review['Metareview'])

    # Initialize the summarizer pipeline
    summarizer = pipeline("summarization",device_id = 0)

    # Prepare the result string
    result = 'Below are multiple summaries of a paper\'s reviews. '
    # Iterate through each paper and its reviews to generate summaries
    if not sample_size:
        sample_size = len(data_list)
    for paper in data_list[:sample_size]:
        for review in paper['ReviewList']:
            summary = summarizer(review, max_length=130,
                                 min_length=60, do_sample=False)
            result += summary[0]['summary_text']+'\n'
        final = summarizer(result, max_length=200,
                           min_length=90, do_sample=False)
        output.append(final[0]['summary_text'])
    return output, gold_metareview
