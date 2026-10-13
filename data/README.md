# Data Layout

This project uses two layers of task data.

- `data/raw/`: original or near-original dataset files
- `data/processed/`: normalized `jsonl` files consumed by the experiment runner

## Closed-Ended Datasets

### CommonsenseQA

Expected raw format: official JSONL with fields such as `id`, `question.stem`, `question.choices`, and `answerKey`.

Example conversion:

```bash
python3 scripts/prepare_closed_ended.py \
  --dataset commonsenseqa \
  --input data/raw/train_rand_split.jsonl \
  --output data/processed/commonsenseqa_train.jsonl
```

### StrategyQA

Expected raw format: JSON list with fields such as `qid`, `question`, and `answer`.

Example conversion:

```bash
python3 scripts/prepare_closed_ended.py \
  --dataset strategyqa \
  --input data/raw/strategyqa_train.json \
  --output data/processed/strategyqa_train.jsonl
```

## Notes

- The normalized schema keeps `task_id`, `question`, `gold_answer`, `choices`, and `metadata`.
- CommonsenseQA keeps its original option set.
- StrategyQA is normalized to binary choices: `yes` / `no`.
- HotpotQA-style integration tasks should go through a separate shard-building pipeline later.
