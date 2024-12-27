import statistics
import logging
import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


def bert_score(refs, outputs):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=outputs, references=refs)
    return results


def bleu_score(ref, pred):
    bleu = load("bleu")
    results = bleu.compute(predictions=pred, references=ref)
    return results


def rouge(ref, output):
    f1_score_map = {}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, output)
    for k, v in scores.items():
        f1_score_map[k] = v.fmeasure
    return f1_score_map


def compute_metrics(tmp_df, ref_col='gpt4_output', output_col='llama3_output', get_all_scores=False):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_inputs = []
    bleu_refs = []
    tmp_df.fillna('', inplace=True)

    for i, row in tmp_df.iterrows():
        ref = row[ref_col]
        output = row[output_col]

        r_score = rouge(ref, output)
        # Get sub-scores
        rouge1 = r_score['rouge1']
        rouge2 = r_score['rouge2']
        rougeL = r_score['rougeL']

        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)

        # Save pairs for Bleu calculation
        bleu_inputs.append(output)
        bleu_refs.append([ref])

    # Compute average scores
    avg_rouge1 = statistics.mean(rouge1_scores)

    avg_rouge2 = statistics.mean(rouge2_scores)

    avg_rougeL = statistics.mean(rougeL_scores)

    # Get Bertscore
    outputs = tmp_df[output_col].values
    refs = tmp_df[ref_col].values
    bertscore = bert_score(refs, outputs)
    bert_f1s = bertscore['f1']
    avg_bertscore = statistics.mean(bert_f1s)

    # Compute BLEU
    avg_bleu = bleu_score(bleu_refs, bleu_inputs)

    metrics = {'ROUGE1': avg_rouge1, 'ROUGE2': avg_rouge2, 'ROUGEL': avg_rougeL, 'BertScore': avg_bertscore, 'BLEU': avg_bleu['bleu'], 'BLEU_JSON': avg_bleu}
    
    if get_all_scores:
        score_df = pd.DataFrame({'ROUGE1': rouge1_scores, 
                            'ROUGE2': rouge2_scores,  
                            'ROUGEL': rougeL_scores,
                            'BertScore': bert_f1s})
        score_df = score_df.melt(var_name='metric', value_name='score')
        score_df = pd.concat([score_df, pd.DataFrame({'metric': ['BLEU'], 'score': [avg_bleu['bleu']]})])
        score_df['model'] = output_col
        
        return metrics, score_df

    return metrics