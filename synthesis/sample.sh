OUTPUT_DIR=sampling_results

if [ ! -e $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

RUN_LOGS=run_logs

if [ ! -e ${RUN_LOGS} ]; then
    mkdir ${RUN_LOGS}
fi

VLLM_PORT=8000

seed=45

pids=$(lsof -ti:$VLLM_PORT)
if [ -n "$pids" ]; then
    echo "Killing ${pids} on port ${VLLM_PORT}"
    echo ${pids} | xargs kill -9
fi

CUDA_VISIBLE_DEVICES="0" python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --max-model-len 32768 \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name ${GENERATOR_NAME} \
        &> "${RUN_LOGS}/${GENERATOR_NAME}_vllm_port_${VLLM_PORT}_seed_${seed}.out" &

python sample.py \
    --output_dir $OUTPUT_DIR \
    --task_type ${TASK_TYPE} \
    --knowledge_base ${KNOWLEDGE_BASE} \
    --seed ${seed} \
    --other_suffix _seed_${seed} \
    --train_set_size ${TRAIN_SET_SIZE} \
    --generator_name ${GENERATOR_NAME} \
    --solver_num_sampling_sequences 32 \
    --vllm_port $VLLM_PORT \
    --num_proc 16 \
    &> ${RUN_LOGS}/sample_${GENERATOR_NAME}_${TASK_TYPE}_seed_${seed}.out