# Python Coding Assistant - RL Experiment

Educational experiment for training a small language model
to improve basic Python coding ability using reinforcement
learning.

## Model

Default model:

Qwen/Qwen2.5-0.5B-Instruct

## Training

Transformers
TRL
GRPO
PyTorch

## Training Environment

Google Colab GPU.

## Local Development

The repository is developed and versioned locally using Git.

No remote Git repository is required.

## Architecture

Text
  |
  v
Tokenizer
  |
  v
Token IDs
  |
  v
Qwen
  |
  v
Generated Token IDs
  |
  +----------------------+
  |                      |
  v                      v
RL computation        decode()
  |                      |
  |                      v
  |                 Human text
  |                      |
  |                      v
  |                Reward function
  |                      |
  +----------+-----------+
             |
             v
          GRPO loss
             |
             v
       Backpropagation
             |
             v
        Model update

## Running

Install dependencies:

pip install -r requirements.txt

Inspect tokenizer:

python scripts/inspect_tokenizer.py \
    --text "def add(a, b): return a + b"

Run training:

python scripts/train.py \
    --config configs/qwen2_5_0_5b_grpo.yaml




#RL repo
python-coding-agent-rl/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── qwen2_5_0_5b_grpo.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── tokenizer.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   │
│   ├── reward/
│   │   ├── __init__.py
│   │   └── reward_function.py
│   │
│   ├── observation/
│   │   ├── __init__.py
│   │   ├── token_viewer.py
│   │   └── generation_viewer.py
│   │
│   └── training/
│       ├── __init__.py
│       ├── trainer.py
│       └── callbacks.py
│
├── scripts/
│   ├── inspect_tokenizer.py
│   ├── test_generation.py
│   └── train.py
│
├── outputs/
│   ├── checkpoints/
│   ├── generations/
│   └── logs/
│
└── notebooks/
    └── 01_training_experiment.ipynb



#Data repo
python-coding-agent-data/
│
├── README.md
│
├── python/
│   ├── syntax/
│   │   ├── variables.jsonl
│   │   ├── data_types.jsonl
│   │   ├── operators.jsonl
│   │   ├── conditions.jsonl
│   │   ├── loops.jsonl
│   │   ├── functions.jsonl
│   │   ├── classes.jsonl
│   │   └── exceptions.jsonl
│   │
│   ├── coding_tasks/
│   │   ├── beginner.jsonl
│   │   ├── intermediate.jsonl
│   │   └── debugging.jsonl
│   │
│   └── evaluation/
│       └── test.jsonl
│
└── metadata/
    └── dataset_info.json