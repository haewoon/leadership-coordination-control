from src.tasks.alphanli import load_alphanli
from src.tasks.commonsenseqa import load_commonsenseqa
from src.tasks.ethics_commonsense import load_ethics_commonsense
from src.tasks.gsm8k import load_gsm8k
from src.tasks.integration_task import load_integration_tasks
from src.tasks.math500 import load_math500
from src.tasks.moralstories import load_moralstories
from src.tasks.mixed_benchmark import load_mixed_benchmark
from src.tasks.social_chemistry import load_social_chemistry
from src.tasks.socialiqa import load_socialiqa
from src.tasks.strategyqa import load_strategyqa

TASK_REGISTRY = {
    "alphanli": load_alphanli,
    "commonsenseqa": load_commonsenseqa,
    "ethics_commonsense": load_ethics_commonsense,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "moralstories": load_moralstories,
    "mixed_benchmark": load_mixed_benchmark,
    "social_chemistry": load_social_chemistry,
    "strategyqa": load_strategyqa,
    "socialiqa": load_socialiqa,
    "integration_task": load_integration_tasks,
}
