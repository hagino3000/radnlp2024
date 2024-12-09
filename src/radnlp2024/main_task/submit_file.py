from radnlp2024.repository import MainTaskResultRepository


def compose_submit_file(name: str):
    MainTaskResultRepository(name).compose_submit_file()
    pass


if __name__ == "__main__":
    compose_submit_file("baseline_gemini_pro")
