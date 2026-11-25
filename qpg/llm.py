"""Model factory."""
from langchain_openai import ChatOpenAI

from qpg import config


def generator_llm():
    # Slight creativity for varied questions, but still controlled.
    return ChatOpenAI(model=config.GENERATOR_MODEL, temperature=0.4)


def judge_llm():
    # Judges and graders must be consistent: temperature 0.
    return ChatOpenAI(model=config.JUDGE_MODEL, temperature=0)
