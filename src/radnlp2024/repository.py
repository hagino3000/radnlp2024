import json
from dataclasses import dataclass
from pathlib import Path

from radnlp2024.models import MainTaskResult

BASE_DIR = Path("dist/main_task")


@dataclass
class MainTaskResultRepository:
    experiment_name: str

    def resolve_save_dir(self) -> Path:
        return BASE_DIR / self.experiment_name

    def exists(self, record_id: str) -> bool:
        save_dir = self.resolve_save_dir()
        return (save_dir / ("ret_" + record_id + ".json")).exists()

    def save(self, record_id: str, result: MainTaskResult) -> None:
        save_dir = self.resolve_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / ("ret_" + record_id + ".json")
        save_file.write_text(result.json())

    def save_prompt(self, record_id: str, prompt: str) -> None:
        save_dir = self.resolve_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / ("prompt_" + record_id + ".txt")
        save_file.write_text(prompt)

    def compose_submission_file(self) -> None:
        save_dir = self.resolve_save_dir()
        submit_file = save_dir / "submission.csv"
        with submit_file.open("w") as f:
            f.write("id,t,n,m\n")
            for file in save_dir.glob("ret_*.json"):
                record_id = file.stem.replace("ret_", "")
                r = json.loads(file.read_text())
                f.write(f"{r['record_id']},{r['t']},{r['n']},{r['m']}\n")
        print(f"Submit file is saved to {submit_file}")
