# Bootstrapping Post-training Signals for Open-ended Tasks via Rubric-based Self-play on Pre-training Text

## Setup

### Clone the repository.

```
git clone https://github.com/HCY123902/POP.git
```

### Install Python environment.

```
uv venv pop --python 3.10.16

source pop/bin/activate

cd POP/

uv pip install -r requirements.txt

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

uv pip install flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Dataset Synthesis

Our experiments are run on 32 CPU cores, 192G memories, and 1 A100 80G GPU.

### Sampling

```
cd POP/synthesis/

bash sample_{task}.sh # Replace {task} with the task name (health_care; creative_writing; instruction_following).
```

Sampling 4096 examples should take around 20 hours to finish.

To use a different reference model, change MODEL_PATH and GENERATOR_NAME in `sample_{task}.sh`.

### Filtering and Pairing

```
cd POP/synthesis/

python filter_and_pair.py --src_path sampling_results/qwen25_7b_seed_45_{task}_ssss_dpo.jsonl # Replace {task} with the task name.
```

If are using a different reference model, change `src_path` accordingly.

### Training

Sign in to WandB.

```
wandb login
```

DPO training.

```
cd POP/training

bash run_dpo_{task}.sh # Replace {task} with the task name.
```

The trained models will be saved in `outputs`.

If are using a different reference model, change `model_name_or_path`, `dataset_splits` and other run name related fields in the `{task}.yaml` file in `training_configs`.

## Trained Checkpoints

| **Model** | **Task** | **Download** |
| :------------: | :------------: | :------------: |
| Qwen2.5-7B | Healthcare QA | [🤗 HuggingFace](https://huggingface.co/POP-Cornell/qwen25_7b_base_hc)   |
| Qwen2.5-7B-Instruct | Healthcare QA | [🤗 HuggingFace](https://huggingface.co/POP-Cornell/qwen25_7b_inst_hc)   |
| Qwen2.5-7B  | Creative Writing | [🤗 HuggingFace](https://huggingface.co/POP-Cornell/qwen25_7b_base_cw)   |
| Qwen2.5-7B-Instruct  | Creative Writing | [🤗 HuggingFace](https://huggingface.co/POP-Cornell/qwen25_7b_inst_cw)   |
| Qwen2.5-7B  | Instruction Following | [🤗 HuggingFace](https://huggingface.co/POP-Cornell/qwen25_7b_base_if)   |
| Qwen2.5-7B-Instruct  | Instruction Following | [🤗 HuggingFace](https://huggingface.co/POP-Cornell/qwen25_7b_inst_if)   |

## Citation

```
@article{huang2026,
    author = {Chengyu Huang and Sheng-Yen Chou and Zhengxin Zhang and Claire Cardie},
    title = {Bootstrapping Post-training Signals for Open-ended Tasks via Rubric-based Self-play on Pre-training Text},
    journal = {arXiv preprint arXiv:2604.20051},
    year = {2026}
}
```