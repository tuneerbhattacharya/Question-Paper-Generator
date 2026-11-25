"""Central settings: models, retrieval sizes and safety limits."""
import os

# Models (cheap, fast defaults; override with environment variables).
GENERATOR_MODEL = os.getenv("QPG_GENERATOR_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("QPG_JUDGE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("QPG_EMBEDDING_MODEL", "text-embedding-3-small")

# Ingestion
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 120

# Retrieval and token budget
MAX_TOPICS = 6            # topics the planner may choose
CHUNKS_PER_TOPIC = 3      # top-k per topic query
MAX_CONTEXT_CHARS = 14_000  # hard cap on context sent to the generator (~3.5k tokens)

# Agent loop limits (stop runaway loops and cost)
MAX_RETRIEVAL_RETRIES = 2   # query rewrites for weakly supported topics
MAX_REVISIONS = 2           # critique -> revise cycles
RECURSION_LIMIT = 30        # LangGraph hard stop on total steps

# Input guardrails
MAX_PDF_MB = 20
MAX_PDF_PAGES = 300
MIN_EXTRACTED_CHARS = 500   # below this the PDF is probably scanned images
MAX_INSTRUCTION_CHARS = 600
MAX_TOTAL_QUESTIONS = 60
