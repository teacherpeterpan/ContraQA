import json

from collections import defaultdict

from tqdm import tqdm

def analyze_jsonl_corpus(file_path):
    count = 0
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            if data['id'].find('-FAKE') >= 0:
                count += 1
    return count

if __name__ == '__main__':
    file_path = 'corpus.json'
    count = analyze_jsonl_corpus(file_path)
    print(count)