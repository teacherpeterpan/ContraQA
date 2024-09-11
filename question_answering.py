import argparse
import json

from nltk.data import retrieve
from reader import MRCPipeline
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Map model name to [model, tokenizer]
MODEL_MAP = {
    'roberta-base': 'deepset/roberta-base-squad2',
    'roberta-large': 'deepset/roberta-large-squad2', 
    'spanbert-large': 'mrm8488/spanbert-finetuned-squadv2',
    'roberta-large-primeQA': 'PrimeQA/squad-v1-roberta-large'
    # 'luke-large': ['studio-ousia/luke-large', 'studio-ousia/luke-large']
}

class SQuAD_question_answering:
    def __init__(self, args) -> None:
        # init QA model
        print(f'loading qa model: {args.qa_model}')
        self.args = args
        if args.qa_model not in MODEL_MAP:
            raise NotImplementedError
        model_name = MODEL_MAP[args.qa_model]
        self.squad_reader = MRCPipeline(model_name)

    def run_qa_for_squad(self):
        with open(os.path.join(self.args.retrieval_results_path), 'r') as f:
            test_dataset = json.load(f)
        
        qa_results = []
        for sample in tqdm(test_dataset):
            ID, question, retrieved_results = sample['id'], sample['question'], sample['retrieval_results']
            correct_answer = [text for text in sample['answers']['text']]
            # run qa model
            contexts = [p['text'] for p in retrieved_results[:self.args.top_k]]
            answers = self.squad_reader.predict(question, contexts)
            out_answers = [{'answer': ans['span_answer_text'], 'score': ans['confidence_score']} for ans in answers]
            # save results
            qa_results.append({'ID': ID, 'question': question, 'answer': correct_answer, 'qa_results': out_answers})
        
        with open(os.path.join(self.args.output_dir, f'qa_{args.qa_model}.json'), 'w') as f:
            f.write(json.dumps(qa_results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_path", type=str, required=True, 
                        help="the path of the retrieval results for the squad dev set")
    parser.add_argument("--qa_model", default=None, type=str, required=True, 
                        help="QA model to use. [bert | roberta-base | roberta-large | spanbert-large]")
    parser.add_argument("--top_k", default=5, type=int, 
                        help="consider the top k retrieved results for QA")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="the path to output QA results")
    args = parser.parse_args()

    QA_model = SQuAD_question_answering(args)
    QA_model.run_qa_for_squad()
