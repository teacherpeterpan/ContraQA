# ContraQA
Data and Codes for ["Attacking Open-domain Question Answering by Injecting Misinformation"](https://arxiv.org/abs/2110.07803) (IJCNLP-AACL 2023; Area Chair Award (Question Answering)). 

Authors: **Liangming Pan, Wenhu Chen, Min-Yen Kan, William Yang Wang**. 

## Introduction 

With a rise in false, inaccurate, and misleading information in propaganda, news, and social media, real-world Question Answering (QA) systems face the challenges of synthesizing and reasoning over misinformation-polluted contexts to derive correct answers. This urgency gives rise to the need to make QA systems robust to misinformation, a topic previously unexplored. We study the risk of misinformation to QA models by investigating the sensitivity of open-domain QA models to corpus pollution with misinformation documents. We curate both human-written and model-generated false documents that we inject into the evidence corpus of QA models and assess the impact on the performance of these systems. Experiments show that QA models are vulnerable to even small amounts of evidence contamination brought by misinformation, with large absolute performance drops on all models. Misinformation attack brings more threat when fake documents are produced at scale by neural models or the attacker targets hacking specific questions of interest. To defend against such a threat, we discuss the necessity of building a misinformation-aware QA system that integrates question-answering and misinformation detection in a joint fashion.

## Environment Setup

- 3.7.0 >= Python < 3.11.0 is required

- Install PyTorch 1.13.0 with CUDA 11.3

```bash
pip install 'torch~=1.13.0' --extra-index-url https://download.pytorch.org/whl/cu113
```

- We are using [PrimeQA](https://github.com/primeqa/primeqa) as for the QA model implemenation. Add primeqa to the PYTHONPATH.

```bash
export PYTHONPATH="./primeqa:$PYTHONPATH"
```

- Install the required packages

```bash
pip install faiss-cpu~=1.7.2
pip install faiss-gpu~=1.7.2
pip install pyserini~=0.20.0
pip install transformers~=4.24.0
pip install datasets[apache-beam]~=2.3.2
```

## Pulloted Corpus with Misinformation

We release the 1M Wikipedia paragraphs with misinformation (*Polluted-Targeted*) as well as the 1M clean Wikipedia paragraphs (*Clean*) in the `./corpus` folder. 

**Polluted-Hybrid**: The 1M Wikipedia paragraphs polluted with misinformation (`wiki_noisy_1M/corpus.json`). Each paragraph is a Dict object with the following fields:

```python
{
    'id': 'Unique ID of the Wikipedia paragraph',
    'contents': 'Text of the Wikipedia paragraph'
}
```

The id with the postfix `-FAKE` indicates that the paragraph is a fake paragraph, for example: `1020959:::Super_Bowl_50-FAKE`.

### Indexing the corpus

To run retrieval-based QA models on the corpus, we need to first index the corpus. We provide the indexing script in `index_corpus.sh`. Run the following command to index the corpus:

```bash
bash index_corpus.sh <corpus_name [wiki_noisy_1M | wiki_clean_1M]>
```

**Clean**: The 1M clean Wikipedia paragraphs (`wiki_clean_1M/corpus.json`). Each paragraph is a Dict object with the same fields as the *Polluted-Hybrid*. 

## QA Under Misinformation Attack

To evaluate the QA model's performance under misinformation attack, run the `run_odqa.sh` script as follows:

```bash
bash run_odqa.sh <device_number> <corpus_name [wiki_noisy_1M | wiki_clean_1M]> <qa_model_name>
```

The results are saved into the `./experiments/` folder. Call `evaluate.py` to evaluate the results. 

## Reference
Please cite the paper in the following format if you use this code during your research.

```
@inproceedings{pan-etal-2023-attacking,
  title = {Attacking Open-domain Question Answering by Injecting Misinformation},
  author = {Pan, Liangming and Chen, Wenhu and Kan, Min-Yen and Wang, William Yang},
  booktitle = {International Joint Conference on Natural Language Processing and Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL)},
  year = {2023},
  address = {Nusa Dua, Bali},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2023.ijcnlp-main.35},
  pages = {525--539}
}
```

## Q&A
If you encounter any problem, please either directly contact the [Liangming Pan](peterpan10211020@gmail.com) or leave an issue in the github repo.