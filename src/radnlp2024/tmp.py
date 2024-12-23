from pathlib import Path
import csv

def main():
    rows = []
    for p in Path("./dataset/radnlp_2024_train_val/ja/main_task/val").glob("*.txt"):
        rows.append([p.stem, p.read_text()])
    
    with Path("./val_dataset.csv").open("w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    main()