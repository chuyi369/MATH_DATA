#export CUDA_VISIBLE_DEVICES=0
BASEMODEL=llama-7b
LORAPATH=llama-7b_MATH-ChatGPT
OUTPATH=sint
python infer.py \
    --base_model  $BASEMODEL \
    --lora_weights $LORAPATH \
    --data_path MATH_data/Raw/test/algebra.json \
    --load_8bit
