CORPUS_NAME=$1

echo 'indexing corpus...'
python primeqa/ir/run_ir.py \
    --do_index \
    --engine_type BM25 \
    --corpus_path ./corpus/$CORPUS_NAME/corpus.json \
    --index_path ./corpus/$CORPUS_NAME/index \
    --threads 24