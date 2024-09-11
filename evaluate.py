import json
from tqdm import tqdm
import random

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))
    # return prediction == truth

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_sample(prediction, gold_answers):
    em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
    f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
    #em_score = compute_exact_match(prediction, gold_answers[0])
    #f1_score = compute_f1(prediction, gold_answers[0])

    return em_score, f1_score

def evaluate_open_domain_QA(result_file, num_of_candidate):
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    total_em = 0.0
    total_f1 = 0.0
    count = 0
    for sample in QA_results:
        gold_answers = sample['answer']

        confs = [tmp['score'] for ind, tmp in enumerate(sample['qa_results']) if ind <= num_of_candidate]
        answers = [tmp['answer'] for ind, tmp in enumerate(sample['qa_results']) if ind <= num_of_candidate]

        max_score_ind = argmax(confs)
        prediction = answers[max_score_ind]

        em_score, f1_score = evaluate_sample(prediction, gold_answers)
        total_em += em_score
        total_f1 += f1_score
        count += 1
    
    avg_em = total_em / count
    avg_f1 = 100 * total_f1 / count
    print(f"EM: {avg_em} \t F1: {avg_f1}")

if __name__ == "__main__":
    result_file = './experiments/wiki_noisy_1M-BM25/qa_roberta-large-primeQA.json'
    evaluate_open_domain_QA(result_file, num_of_candidate = 5)