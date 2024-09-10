# ContraQA
Data and Codes for ["Attacking Open-domain Question Answering by Injecting Misinformation"](https://arxiv.org/abs/2110.07803) (IJCNLP-AACL 2023; Area Chair Award (Question Answering)). 

Authors: **Liangming Pan, Wenhu Chen, Min-Yen Kan, William Yang Wang**. 

## Introduction 

With a rise in false, inaccurate, and misleading information in propaganda, news, and social media, real-world Question Answering (QA) systems face the challenges of synthesizing and reasoning over misinformation-polluted contexts to derive correct answers. This urgency gives rise to the need to make QA systems robust to misinformation, a topic previously unexplored. We study the risk of misinformation to QA models by investigating the sensitivity of open-domain QA models to corpus pollution with misinformation documents. We curate both human-written and model-generated false documents that we inject into the evidence corpus of QA models and assess the impact on the performance of these systems. Experiments show that QA models are vulnerable to even small amounts of evidence contamination brought by misinformation, with large absolute performance drops on all models. Misinformation attack brings more threat when fake documents are produced at scale by neural models or the attacker targets hacking specific questions of interest. To defend against such a threat, we discuss the necessity of building a misinformation-aware QA system that integrates question-answering and misinformation detection in a joint fashion.

## Pulloted Corpus with Misinformation

We release the 1M Wikipedia paragraphs with misinformation (*Polluted-Hybrid*) as well as the 1M clean Wikipedia paragraphs (*Clean*) in the `./corpus` folder. 

**Polluted-Hybrid**: The 1M Wikipedia paragraphs polluted with misinformation (`noisy_wiki_1M/corpus.jsonl`). Each paragraph is a Dict object with the following fields:

```python
{
    '_id': 'Unique ID of the Wikipedia paragraph',
    'title': 'Title and type of the Wikipedia paragraph',
    'text': 'Text of the Wikipedia paragraph',
    'metadata': 'Metadata of the Wikipedia paragraph'
}
```

The `title` filed is in the format of `<paragraph_title>:::<paragraph_type>`, where `<paragraph_title>` is the title of the Wikipedia paragraph and `<paragraph_type>` is the type of the paragraph. For example, the type `FAKE_Human` indicates that the paragraph is a human-annotated fake paragraph. 

**Clean**: The 1M clean Wikipedia paragraphs (`clean_wiki_1M/corpus.jsonl`). Each paragraph is a Dict object with the same fields as the *Polluted-Hybrid*. 

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