import json
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from radnlp2024.genai import GenerativeModel
from radnlp2024.main_task.baseline.process import classify_lung_cancer_staging
from radnlp2024.models import M_category, MainTaskResult, N_category, T_category
from radnlp2024.repository import MainTaskResultRepository


def main(name):
    logger.info("Start main task baseline")
    data_dir = Path("dataset/radnlp_2024_train_val/ja/main_task")
    train_data_dir = data_dir / "train"
    validate_data_dir = data_dir / "val"
    assert train_data_dir.exists()
    assert validate_data_dir.exists()

    few_short_samples = build_fewshot_samples(train_data_dir)

    repo = MainTaskResultRepository(name)
    for f in tqdm(validate_data_dir.glob("*.txt")):
        if repo.exists(f.stem):
            print(f"Skip. Result of {f.stem} already exists")
            continue
        record_id = f.stem
        logger.info(record_id)
        report = f.read_text().strip()
        ret, prompt = classify_lung_cancer_staging(
            report, few_short_samples, generative_model=GenerativeModel.Gemini15Pro001
        )
        repo.save(record_id, MainTaskResult(record_id, ret.t, ret.n, ret.m))
        repo.save_prompt(record_id, prompt)
        time.sleep(20)

    logger.info("Finish main task baseline")


def load_labels(path: Path) -> dict[str, MainTaskResult]:
    rows = path.read_text().strip().split("\n")
    ret = {}
    for row in rows[1:]:
        key, t, n, m = row.split(",")
        ret[key] = MainTaskResult(key, T_category(t), N_category(n), M_category(m))
    return ret


def build_fewshot_samples(train_data_dir: Path) -> str:
    train_data_labels = load_labels(train_data_dir / "label.csv")
    assert train_data_dir.exists()

    few_shot_lines = []
    for f in train_data_dir.glob("*.txt"):
        labels = train_data_labels[f.stem]
        few_shot_lines.append("input:")
        few_shot_lines.append(f.read_text().strip())
        few_shot_lines.append("\n")
        few_shot_lines.append("output:")
        few_shot_lines.append(json.dumps({"t": labels.t.name, "n": labels.n.name, "m": labels.m.name}))
        few_shot_lines.append("\n")

    few_shot_samples_text = "\n".join(few_shot_lines)
    return few_shot_samples_text


if __name__ == "__main__":
    main("baseline_gemini_pro")
