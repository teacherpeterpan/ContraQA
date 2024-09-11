# the scripts to run ODQA experiments
export CUDA_VISIBLE_DEVICES=$1

CORPUS_NAME=$2
IR_METHOD=BM25
READER=$3

SETTING=$CORPUS_NAME-$IR_METHOD
mkdir -p ./experiments/$SETTING

echo 'BM25 retrieval...'
python sparse_retriever.py \
    --query_path ./corpus/$CORPUS_NAME/queries.json \
    --index_path ./corpus/$CORPUS_NAME/index \
    --top_k 50 \
    --output_dir ./experiments/$SETTING

echo 'Question answering...'
python question_answering.py \
    --retrieval_results_path ./experiments/$SETTING/retrieval_results.json \
    --qa_model roberta-large-primeQA \
    --top_k 5 \
    --output_dir ./experiments/$SETTING