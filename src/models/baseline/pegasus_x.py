from transformers import AutoTokenizer, PegasusForConditionalGeneration
import torch
def run_pegasus_x_summarization(sample_size:int, data_list:list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PegasusForConditionalGeneration.from_pretrained(
    "google/pegasus-x-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")
    output = []
    gold_metareview = []
    for review in data_list[:sample_size]:
        gold_metareview.append(review['Metareview'])
    result = 'Below are multiple summaries of a paper\'s reviews. '
    if not sample_size:
        sample_size = len(data_list)
    for paper in data_list[:sample_size]:
        for review in paper['ReviewList']:
            inputs = tokenizer(review,
                            max_length=500, return_tensors="pt")
            summary_ids = model.generate(inputs["input_ids"])
            summary = tokenizer.batch_decode(
                summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result += summary+'\n'
        final = tokenizer(result,
                        max_length=500, return_tensors="pt")
        meta_summary_ids = model.generate(final["input_ids"])
        meta_sum = tokenizer.batch_decode(
            meta_summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output.append(meta_sum)
    return output, gold_metareview