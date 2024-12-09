import os
from enum import Enum
from typing import Literal

import litellm
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]
Message = tuple[Role, str]

load_dotenv()  # load OPENAI_API_KEY, VERTEX_AI_PROJECT, VERTEX_AI_LOCATION
litellm.enable_json_schema_validation = True
litellm.vertex_project = os.getenv("VERTEX_AI_PROJECT")
litellm.vertex_location = os.getenv("VERTEX_AI_LOCATION")
litellm.vertex_ai_safety_settings = [
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "OFF",
    },
]


class GenerativeModel(Enum):
    GPT4oMini = "gpt-4o-mini"
    GPT4o = "gpt-4o"
    Gemini15Pro001 = "vertex_ai/gemini-1.5-pro-001"
    Gemini15Pro002 = "vertex_ai/gemini-1.5-pro-002"
    Gemini15Flash001 = "vertex_ai/gemini-1.5-flash-001"
    Gemini15Flash002 = "vertex_ai/gemini-1.5-flash-002"


def text_completion(
    generative_model: GenerativeModel,
    messages: list[Message],
    response_format: BaseModel | dict | None = None,
    temperature: float = 0.1,
) -> tuple[bool, str, str]:
    response = litellm.completion(
        model=generative_model.value,
        messages=[{"role": role, "content": message} for role, message in messages],
        response_format=response_format,
        temperature=temperature,
    )

    usage = response.usage
    total_input_length = sum([len(message) for _, message in messages])
    logger.debug(
        f"GenAI Usage. Input length:{total_input_length}, Prompt Token:{usage.prompt_tokens}, Completion Token:{usage.completion_tokens}"
    )
    output = response.choices[0]
    success = output.finish_reason == "stop"
    return success, output.message.content, output.finish_reason


def test():
    m = GenerativeModel.Gemini15Flash001
    a, b, c = text_completion(m, [("user", "Hello")], temperature=1)
    print(f"model: {m.value}, success: {a}, response: {b.rstrip()}, StopReason: {c}")

    m = GenerativeModel.GPT4oMini
    a, b, c = text_completion(m, [("user", "こんにちは")], temperature=1)
    print(f"model: {m.value}, success: {a}, response: {b.rstrip()}, StopReason: {c}")


if __name__ == "__main__":
    test()
