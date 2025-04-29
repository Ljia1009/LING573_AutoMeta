import pandas as pd
import glob
<<<<<<< HEAD
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt') 


file_paths = glob.glob("../evaluation/metrics/*.csv")
model_results = {}

for file_path in file_paths:
    model_name = file_path.split("/")[-1].replace(".txt.csv", "")
=======
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

file_paths = glob.glob("../evaluation/*.csv")

regular_files = [fp for fp in file_paths if not os.path.basename(fp).startswith('disco')]
main_metrics = []

for file_path in regular_files:
    model_name = os.path.basename(file_path).replace("_out.txt.csv", "")
>>>>>>> 81595ee (add analysis results)
    df = pd.read_csv(file_path)

    df['prediction_tokens'] = df['prediction'].apply(lambda x: len(word_tokenize(str(x))))

<<<<<<< HEAD
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
=======
    summary = {
        'model': model_name,
        'Prediction-Tokens-Length': df['prediction_tokens'].mean(),
        'ROUGE': df['rouge_score'].mean(),
        'BERTScore-F1': df['bertscore_f1'].mean(),
        'BERTScore-precision': df['bertscore_precision'].mean(),
        'BERTScore-recall': df['bertscore_recall'].mean(),
        'FactCC-Score': df['factCC_score'].mean(),
        'FactCC-Label-Rate': (df['factCC_label'] == 'CORRECT').mean(),
    }
    main_metrics.append(summary)

df_main_metrics = pd.DataFrame(main_metrics)

disco_files = [fp for fp in file_paths if os.path.basename(fp).startswith("disco")]

disco_metrics = []

for file_path in disco_files:
    model_name = os.path.basename(file_path).replace("disco_", "").replace("_out.txt.csv", "")
    df = pd.read_csv(file_path)

    summary = {
        'model': model_name,
        'Disco_EntityGraph': df['EntityGraph'].mean(),
        'Disco_LexicalChain': df['LexicalChain'].mean(),
        'Disco_RC': df['RC'].mean(),
        'Disco_LC': df['LC'].mean(),
    }
    disco_metrics.append(summary)

df_disco_metrics = pd.DataFrame(disco_metrics)

df_merged = pd.merge(df_main_metrics, df_disco_metrics, on='model', how='outer')  # outer in case some models are missing either part

df_merged = df_merged.round(4)

df_merged = df_merged.sort_values(by="model", ascending=False)
df_merged.to_csv("../evaluation/analysis/evaluation_summary_combined.csv", index=False)

print(df_merged)
>>>>>>> 81595ee (add analysis results)
