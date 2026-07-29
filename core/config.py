"""
core/config.py
Central configuration: paths, API keys, LLM provider factory.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

# ---------------------------------------------------------------------------
# Data directories (medallion layers + supporting artefacts)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data"

BRONZE_DIR = BASE_DIR / "bronze"
SILVER_DIR = BASE_DIR / "silver"
GOLD_DIR = BASE_DIR / "gold"
STTM_DIR = BASE_DIR / "sttm"
PROFILES_DIR = BASE_DIR / "profiles"
REPORTS_DIR = BASE_DIR / "reports"
TRACES_DIR = BASE_DIR / "traces"
AUDIT_DIR = PROJECT_ROOT / "audit_logs"
UPLOADS_DIR = BASE_DIR / "uploads"

for _d in (
    BRONZE_DIR, SILVER_DIR, GOLD_DIR, STTM_DIR,
    PROFILES_DIR, REPORTS_DIR, TRACES_DIR, AUDIT_DIR, UPLOADS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# One shared rate limiter for all agents — created once at import time so
# every make_llm() call (across profiler, sttm_generator, reporter, etc.)
# draws from the same token bucket instead of each agent having its own.
# GitHub Models free tier: 10 RPM → 1 request every 10 s to stay safely under.
from langchain_core.rate_limiters import InMemoryRateLimiter as _RLimiter
_GITHUB_RATE_LIMITER = _RLimiter(
    requests_per_second=1 / 15,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)


def make_llm(temperature: float = 0.0):
    """
    Factory that returns a LangChain chat model based on LLM_PROVIDER.
    Every agent calls this instead of instantiating a provider directly,
    so switching providers is a single .env change.
    """
    if LLM_PROVIDER == "github":
        if not GITHUB_TOKEN:
            raise ValueError(
                "GITHUB_TOKEN is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI

        # max_retries uses the OpenAI SDK's built-in exponential backoff on 429s
        # without wrapping in RunnableRetry, so bind_tools() still works.
        return ChatOpenAI(
            model=GITHUB_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_BASE_URL,
            temperature=temperature,
            rate_limiter=_GITHUB_RATE_LIMITER,
            max_retries=6,
        )

    if LLM_PROVIDER == "google":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file, "
                "or set LLM_PROVIDER=groq to use Groq instead."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    # default: groq
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file, "
            "or set LLM_PROVIDER=google to use Gemini instead."
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
    )
