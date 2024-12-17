import argparse

from radnlp2024.repository import MainTaskResultRepository


def compose_submission_file(name: str):
    MainTaskResultRepository(name).compose_submission_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose submission file")
    parser.add_argument("name", type=str, help="The name for the submission file")
    args = parser.parse_args()

    compose_submission_file(args.name)
