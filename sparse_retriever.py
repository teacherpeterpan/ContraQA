from primeqa.ir.sparse.retriever import PyseriniRetriever
import pandas as pd
import argparse
import json
import os
from tqdm import tqdm

class SQuAD_sparse_retriever:
    def __init__(self, args) -> None:
        self.args = args
        # Instantiate the retriever
        self.searcher = PyseriniRetriever(args.index_path, use_bm25=True, k1=args.k1, b=args.b)
        # load the questions in the squad dev set
        self.load_squad_queries()
    
    def load_squad_queries(self):
        with open(self.args.query_path, 'r') as f:
            self.squad_dev_set = json.load(f)
    
    '''
    get the retrieval results for the squad dev set
    '''
    def search_squad(self):
        results = []
        for sample in tqdm(self.squad_dev_set):
            ID, query = sample['id'], sample['question']
            hits = self.searcher.retrieve(query, self.args.top_k)
            sample['retrieval_results'] = [{'id': hit['doc_id'], 'text': hit['text'], 'score': hit['score']} for hit in hits]
            results.append(sample)

        with open(os.path.join(args.output_dir, 'retrieval_results.json'), 'w') as f:
            f.write(json.dumps(results, indent=2))

    def search_with_given_queries(self, queries):
        # Run queries
        for query in queries:
            hits = self.searcher.retrieve(query, self.args.top_k)
            df = pd.DataFrame.from_records(hits, columns=['rank','score','doc_id','title','text'])
            print('======================================================================')
            print(f'QUERY: {query}')
            print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, required=True, 
                        help="the path of the squad dev set")
    parser.add_argument("--index_path", type=str, required=True, 
                        help="the path of the corpus index")
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--k1", default=0.9, type=float)
    parser.add_argument("--b", default=0.4, type=float)
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="the path of the output retrival results")
    args = parser.parse_args()

    retriever = SQuAD_sparse_retriever(args)
    retriever.search_squad()

    # for testing
    # queries = [
    #     'Which NFL team represented the AFC at Super Bowl 50?',
    #     'Where did Super Bowl 50 take place?',
    #     'Which NFL team won Super Bowl 50?'
    # ]
    # retriever = SQuAD_sparse_retriever(args)
    # retriever.search_with_given_queries(queries)