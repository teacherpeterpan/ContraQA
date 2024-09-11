from transformers import AutoConfig, AutoTokenizer
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor

from transformers import DataCollatorWithPadding
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers

from primeqa.mrc.trainers.mrc import MRCTrainer
from datasets import Dataset
import json

class MRCPipeline():
    
    def __init__(self, model_for_qa):
        task_heads = EXTRACTIVE_HEAD
        config = AutoConfig.from_pretrained(
            model_for_qa,
            cache_dir = './cache'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_for_qa,
            use_fast=True,
            config=config,
        )

        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = ModelForDownstreamTasks.from_config(
            config,
            model_for_qa,
            task_heads=task_heads,
        )
        model.set_task_head(next(iter(task_heads)))        

        self.preprocessor = BasePreProcessor(
            stride=256,
            max_seq_len=512,
            tokenizer=tokenizer,)
        
        data_collator = DataCollatorWithPadding(tokenizer)
        postprocessor = ExtractivePostProcessor(
            k=10,
            n_best_size=20,
            max_answer_length=200,
            scorer_type=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF,
            single_context_multiple_passages=self.preprocessor._single_context_multiple_passages,
        )
        
        self.trainer = MRCTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocessor.process
        )
        
    def predict(self, question, context):
        questions = [question]
        contexts = [context]
        example_ids = [str(0)]
        
        examples_dict = dict(question=questions, context=contexts, example_id=example_ids)
        eval_examples = Dataset.from_dict(examples_dict)
        
        eval_examples, eval_dataset = self.preprocessor.process_eval(eval_examples)
        predictions = self.trainer.predict(eval_dataset=eval_dataset, eval_examples=eval_examples)
        
        original_answers = list(predictions.values())[0]
        
        processed_answers = []
        for a in original_answers:
            processed_answers.append({'span_answer_text' : a['span_answer_text'],
                                        'confidence_score': a['confidence_score']})
        return processed_answers

if __name__ == "__main__":
    # load a reader
    reader = MRCPipeline("PrimeQA/squad-v1-roberta-large")

    question = "Which NFL team represented the AFC at Super Bowl 50?"
    context = ["Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
    "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2012 season. The American Football Conference (AFC) champion San Francisco 49ers defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201308 to earn their third Super Bowl title. The game was played on February 9, 2012, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as adding roman numerals  (under which the game is also known as \"Super Bowl L\"), and the logo also feature the Arabic numerals 50.",
    "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2012 season. The American Football Conference (AFC) champion California 48ers defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201308 to earn their third Super Bowl title. The game was played on February 9, 2012, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as adding roman numerals  (under which the game is also known as \"Super Bowl L\"), and the logo also feature the Arabic numerals 50."]

    answers = reader.predict(question, context)  
    print(json.dumps(answers, indent=4))