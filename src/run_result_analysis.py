import pandas as pd
import glob
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt') 


file_paths = glob.glob("../evaluation/metrics/*.csv")
model_results = {}

for file_path in file_paths:
    model_name = file_path.split("/")[-1].replace(".txt.csv", "")
    df = pd.read_csv(file_path)

    df['prediction_tokens'] = df['prediction'].apply(lambda x: len(word_tokenize(str(x))))

    results = {
        "ROUGE-Avg": df['rouge_score'].mean(),
        "ROUGE-Std": df['rouge_score'].std(),
        "BERTScore-F1-Avg": df['bertscore_f1'].mean(),
        "BERTScore-F1-Std": df['bertscore_f1'].std(),
        "FactCC-Score-Avg": df['factCC_score'].mean(),
        "FactCC-Score-Std": df['factCC_score'].std(),
        "FactCC-Label-Consistency-Rate": (df['factCC_label'] == 'CORRECT').mean(),
        "Avg-Prediction-Tokens": df['prediction_tokens'].mean(),
    }

    model_results[model_name] = results


comparison_df = pd.DataFrame(model_results).T


comparison_df = comparison_df.sort_values(by="BERTScore-F1-Avg", ascending=False)
comparison_df = comparison_df.round(4)
comparison_df.to_csv("../evaluation/evaluation_summary.csv", index=True)