export WANDB_PROJECT="pop"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="outputs"

if [ ! -e ${OUTPUT_DIR} ]; then
    mkdir ${OUTPUT_DIR}
fi

RUN_LOGS="run_logs"

if [ ! -e ${RUN_LOGS} ]; then
    mkdir ${RUN_LOGS}
fi

CUDA_VISIBLE_DEVICES="0" ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml run_dpo.py training_configs/${RUN_NAME}.yaml &> ${RUN_LOGS}/${RUN_NAME}.out