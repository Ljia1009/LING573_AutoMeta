import sys
from evaluate import load
from transformers import pipeline
#from summac.model_summac import SummaCZS, SummaCConv
#from disco_score import DiscoScorer
sys.path.insert(0,"src/models/baseline")
from bart import output, gold_metareview


class Evaluator:
    def __init__(self, predictions, references):
        self.predictions = predictions
        self.references = references

    def _rouge(self):
        rouge=load('rouge')
        return rouge.compute(predictions=self.predictions,
                         references=self.references,
                         rouge_types=['rougeL'],
                         use_aggregator=False)
    
    def _bertscore(self, model_type):
        bertscore = load("bertscore")
        return bertscore.compute(predictions=self.predictions,
                                references=self.references,
                                model_type=model_type)
    
    def _factCC(self, source_docs, summaries):
        # TODO: FactCC: https://huggingface.co/manueldeprada/FactCC
        # Note: FactCC is not comparing reference and predictions but summaries and source texts
        pipe=pipeline(model="manueldeprada/FactCC")
        data = [[[source_docs[i], summaries[i]]] for i in range(len(source_docs))]
        return pipe(data, truncation='only_first',padding='max_length')


    def _summaC(self, source_docs, summaries):
        # TODO: SummaC:https://github.com/tingofurro/summac
        model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")
        
        scores_zs, scores_conv = [], []
        data = [(source_docs[i], summaries[i]) for i in range(len(source_docs))]
        for doc, summary in data:
            scores_zs.append(model_zs.score([doc], [summary])["score"][0])
            scores_conv.append(model_conv.score([doc], [summary])["score"][0])

        return scores_zs, scores_conv
    
    def _discoScore(self, system, references):
        # TODO: https://github.com/AIPHES/DiscoScore
        # TODO: which discourse metric to use?
        # TODO: not sure what the system means
        disco_scorer = DiscoScorer(device='cuda:0', model_name='bert-base-uncased')

        for s, refs in zip(system, references):
            s = s.lower()
            refs = [r.lower() for r in refs]
            print(disco_scorer.EntityGraph(s, refs))
            print(disco_scorer.LexicalChain(s, refs))
            print(disco_scorer.RC(s, refs))    
            print(disco_scorer.LC(s, refs)) 
            print(disco_scorer.DS_Focus_NN(s, refs)) # FocusDiff 
            print(disco_scorer.DS_SENT_NN(s, refs)) # SentGraph

    def evaluate(self, metric, **kwargs):
        if metric == "rouge_L":
            return self._rouge()
        elif metric == "bertscore":
            return self._bertscore(model_type=kwargs.get("model_type", "distilbert-base-uncased"))
        elif metric == "factCC":
            return self._factCC(
                source_docs=kwargs["source_docs"],
                summaries=kwargs["summaries"],
            )
        elif metric == "summaC":
            return self._summaC(
                source_docs=kwargs["source_docs"],
                summaries=kwargs["summaries"],
            )
        elif metric == "disco":
            return self._discoScore(
                system=kwargs["system"],
                references=kwargs["references"],
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
if __name__ == "__main__":
    # preds = ["hello world", "general kenobi"]
    # refs  = ["goodnight moon", "the sun is shining"]

    ev = Evaluator(output, gold_metareview)

    rouge_scores = ev.evaluate("rouge_L")
    print("ROUGE:", rouge_scores)

    bert_results = ev.evaluate("bertscore", model_type="distilbert-base-uncased")
    print("BERTScore:", bert_results)