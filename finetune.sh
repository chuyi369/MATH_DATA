BASEMODEL=llama-7b
LORAPATH=None
OUTPATH=llama-7b_MATH-ChatGPT
wandb online
export WORLD_SIZE=2
#cd $ROOTPATH
torchrun --nproc_per_node=8 finetune.py \
    --base_model $BASEMODEL \
    --resume_from_checkpoint $LORAPATH \
    --data_path MATH_data/ChatGPT \
    --output_dir $OUTPATH
