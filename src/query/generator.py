# src/query/generator.py
"""
Hybrid generator with DeepSeek V3.2 for everything.
  fact, discovery  → deepseek-chat (fast, non-thinking)
  comparison       → deepseek-reasoner (thinking mode)
  synthesis        → deepseek-reasoner (thinking mode)
  concept extraction → deepseek-reasoner (called directly)

Fallback to Qwen3:14b on any DeepSeek failure (503, timeout).
"""

import os
import re
import time
import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

QWEN_MODEL          = "qwen3:14b"
DEEPSEEK_CHAT       = "deepseek-chat"       # V3.2 non-thinking
DEEPSEEK_REASONER   = "deepseek-reasoner"   # V3.2 thinking mode
DEEPSEEK_TIMEOUT    = 120                   # seconds — DeepSeek can be slow
DEEPSEEK_MAX_RETRY  = 2

_deepseek_client = None


def get_deepseek_client() -> OpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set in .env — "
                "get one at https://platform.deepseek.com"
            )
        _deepseek_client = OpenAI(
            api_key  = api_key,
            base_url = "https://api.deepseek.com",
            timeout  = DEEPSEEK_TIMEOUT,
        )
    return _deepseek_client


# ── Intent → model routing ────────────────────────────────────────────────────

def route_model(intent: str) -> str:
    """
    Returns DeepSeek model string for each intent.
    comparison + synthesis → reasoner (thinking mode)
    fact + discovery       → chat (non-thinking, fast)
    """
    if intent in ("comparison", "synthesis"):
        return DEEPSEEK_REASONER
    return DEEPSEEK_CHAT


# ── Evidence budgets ──────────────────────────────────────────────────────────

EVIDENCE_BUDGETS = {
    "fact":       {"max_chunks": 8,  "max_tokens": 2000},
    "comparison": {"max_chunks": 12, "max_tokens": 4000},
    "synthesis":  {"max_chunks": 15, "max_tokens": 5000},
    "discovery":  {"max_chunks": 10, "max_tokens": 2000},
}


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scientific assistant for astrophysics research \
on intracluster magnetic fields, Faraday rotation, and radio observations \
of galaxy clusters.

Your task is to answer questions using ONLY the provided evidence chunks \
from scientific papers.

Rules:
1. Answer ONLY from the evidence provided. Do not use external knowledge.
2. Cite every claim with [E1], [E2], etc. referencing the evidence numbers.
3. If evidence is insufficient or unclear, say so explicitly.
4. If evidence from different papers contradicts each other, report the \
disagreement — do not hide it.
5. Be precise with numerical values — quote them exactly as they appear.
6. Use LaTeX math for all symbols: $\\sigma_{RM}$, $B_0$, $n = 2$, \
$\\Lambda_{min}$, $\\eta$.
7. Keep your answer focused and grounded in the evidence."""


# ── Evidence formatter ────────────────────────────────────────────────────────

def format_evidence(evidence: list[dict], max_chunks: int) -> str:
    lines = ["=== EVIDENCE ===\n"]
    for i, e in enumerate(evidence[:max_chunks]):
        lines.append(
            f"[E{i+1}] {e.get('title','?')[:55]} ({e.get('year','?')})\n"
            f"       paper   : {e.get('paper_id','')}\n"
            f"       section : {e.get('section','')}\n"
            f"       rerank  : {e.get('rerank_score',0):.3f}\n"
            f"       text    : {e.get('text','')[:400]}\n"
        )
    return "\n".join(lines)


def format_support_stats(stats: dict) -> str:
    lines = ["=== SUPPORT STATS ==="]
    lines.append(f"Distinct papers  : {stats['distinct_papers']}")
    lines.append(f"Max rerank score : {stats['max_rerank_score']}")
    if stats.get("has_disagreement"):
        lines.append("⚠ Disagreement detected between papers")
    return "\n".join(lines)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query: str, pack: dict, intent: str) -> str:
    budget   = EVIDENCE_BUDGETS.get(intent, EVIDENCE_BUDGETS["fact"])
    evidence = format_evidence(pack["evidence"], budget["max_chunks"])
    stats    = format_support_stats(pack["support_stats"])

    if pack.get("abstain"):
        extra = (
            f"\n⚠ ABSTENTION SIGNAL: {pack.get('abstain_reason','insufficient evidence')}\n"
            "Say explicitly: 'The available evidence is insufficient to answer this question.'\n"
        )
    elif pack["support_stats"].get("has_disagreement"):
        extra = (
            "\n⚠ DISAGREEMENT: Papers contradict each other. "
            "Report the disagreement explicitly.\n"
        )
    else:
        extra = ""

    return f"""{stats}

{evidence}
{extra}
=== QUESTION ===
{query}

=== INSTRUCTIONS ===
Answer using only the evidence above. Cite as [E1], [E2], etc.
Use LaTeX math for all symbols and equations.
Quote numerical values exactly as they appear in the evidence."""


# ── Math post-processor ───────────────────────────────────────────────────────

def _fix_math_notation(text: str) -> str:
    """Normalize LLM math output to clean LaTeX."""
    replacements = [
        (r'B\s*[\n\r]+0\b',                r'$B_0$'),
        (r'B\s*_\s*{\s*0\s*}',             r'$B_0$'),
        (r'B\s*_\s*0\b',                   r'$B_0$'),
        (r'⟨B\s*[\n\r]+0\s*⟩',             r'$\langle B_0 \rangle$'),
        (r'⟨B_0⟩',                          r'$\langle B_0 \rangle$'),
        (r'⟨B⟩\s*_?\s*0\b',               r'$\langle B_0 \rangle$'),
        (r'σ\s*_?\s*RM\b',                 r'$\sigma_{RM}$'),
        (r'σ\s*[\n\r]+RM\b',               r'$\sigma_{RM}$'),
        (r'sigma\s*_?\s*RM\b',             r'$\sigma_{RM}$'),
        (r'⟨RM⟩',                           r'$\langle RM \rangle$'),
        (r'\|⟨RM⟩\|',                       r'$|\langle RM \rangle|$'),
        (r'Λ\s*_?\s*min\b',                r'$\Lambda_{min}$'),
        (r'Λ\s*_?\s*max\b',                r'$\Lambda_{max}$'),
        (r'λ\s*_?\s*min\b',                r'$\lambda_{min}$'),
        (r'λ\s*_?\s*max\b',                r'$\lambda_{max}$'),
        (r'\bn\s*=\s*(\d+(?:\.\d+)?)\b',   r'$n = \1$'),
        (r'\bη\s*=\s*([\d.]+)',             r'$\eta = \1$'),
        (r'\bα\s*[∼~≈]\s*([\d.]+)',        r'$\\alpha \\sim \1$'),
        (r'\bα\s*=\s*([\d.]+)',             r'$\\alpha = \1$'),
        (r'(\d+(?:\.\d+)?)\s*µ\s*G\b',    r'$\1\,\mu\mathrm{G}$'),
        (r'(\d+(?:\.\d+)?)\s*μ\s*G\b',    r'$\1\,\mu\mathrm{G}$'),
        (r'(\d+(?:\.\d+)?)\s*rad\s*m[-–]\s*²', r'$\1\,\mathrm{rad\,m^{-2}}$'),
        (r'(\d+(?:\.\d+)?)\s*kpc\b',       r'$\1\,\mathrm{kpc}$'),
        (r'(\d+(?:\.\d+)?)\s*Mpc\b',       r'$\1\,\mathrm{Mpc}$'),
        (r'\$\$([^$\n]+)\$\$',              r'$\1$'),
    ]
    for pattern, replacement in replacements:
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            continue
    return text


# ── Backend generators ────────────────────────────────────────────────────────

def _generate_deepseek(
    prompt: str,
    intent: str,
) -> tuple[str, str | None]:
    """
    Call DeepSeek API with retry on 503.
    Returns (answer, thinking_text or None).
    """
    client = get_deepseek_client()
    model  = route_model(intent)
    is_reasoner = model == DEEPSEEK_REASONER

    for attempt in range(DEEPSEEK_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model    = model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.1,
                max_tokens  = 2000 if not is_reasoner else 4000,
            )
            answer   = response.choices[0].message.content.strip()
            thinking = getattr(
                response.choices[0].message, "reasoning_content", None
            )
            return answer, thinking

        except Exception as e:
            if attempt < DEEPSEEK_MAX_RETRY - 1:
                wait = 5 * (attempt + 1)
                print(f"  ⚠ DeepSeek attempt {attempt+1} failed ({e}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _generate_qwen3(
    prompt: str,
    intent: str,
) -> tuple[str, str | None]:
    """Fallback: local Qwen3:14b via Ollama."""
    response = ollama.chat(
        model    = QWEN_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        think   = False,
        options = {"temperature": 0.1, "num_predict": 1500},
    )
    answer   = response["message"]["content"].strip()
    thinking = response["message"].get("thinking", None)
    return answer, thinking


# ── Citation extractor ────────────────────────────────────────────────────────

def extract_citations(answer: str, evidence: list[dict]) -> list[dict]:
    cited = sorted(set(
        int(m) - 1
        for m in re.findall(r"\[E(\d+)\]", answer)
        if int(m) - 1 < len(evidence)
    ))
    return [evidence[i] for i in cited]


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_answer(
    query:          str,
    pack:           dict,
    synthesis_mode: str = "non_thinking",
) -> dict:
    """
    Generate a grounded answer. Routes all intents through DeepSeek.
    Falls back to Qwen3 on DeepSeek failure.
    """
    intent = pack.get("intent", "fact")
    model  = route_model(intent)

    # Hard abstention
    if pack.get("abstain") and pack.get("abstain_reason") != "disagreement_present":
        return {
            "answer":    (
                "Insufficient evidence to answer this question. "
                f"Reason: {pack.get('abstain_reason', 'unknown')}"
            ),
            "citations": [],
            "abstained": True,
            "thinking":  None,
            "model":     "none",
            "backend":   "none",
            "intent":    intent,
        }

    prompt  = build_prompt(query, pack, intent)
    backend = "deepseek"

    print(f"  Generator: intent={intent} model={model}")

    try:
        answer, thinking = _generate_deepseek(prompt, intent)
        model_used = f"deepseek/{model}"

    except Exception as e:
        print(f"  ⚠ DeepSeek failed ({e}) — falling back to Qwen3")
        answer, thinking = _generate_qwen3(prompt, intent)
        model_used = f"{QWEN_MODEL} (fallback)"
        backend    = "qwen3_fallback"

    answer    = _fix_math_notation(answer)
    citations = extract_citations(answer, pack.get("evidence", []))

    return {
        "answer":    answer,
        "citations": citations,
        "abstained": False,
        "thinking":  thinking,
        "model":     model_used,
        "backend":   backend,
        "intent":    intent,
    }


# ── Direct extraction helper (for concept graph) ──────────────────────────────

def extract_with_deepseek(
    prompt:    str,
    max_tokens: int = 1000,
) -> str:
    """
    Direct DeepSeek call for structured extraction tasks.
    Uses reasoner for quality. Returns raw text.
    """
    client = get_deepseek_client()
    try:
        response = client.chat.completions.create(
            model       = DEEPSEEK_REASONER,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ DeepSeek extraction failed: {e}")
        return ""


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(query: str, result: dict) -> None:
    print(f"\n{'='*65}")
    print(f"QUERY   : {query}")
    print(f"INTENT  : {result['intent']}")
    print(f"BACKEND : {result['backend']} ({result['model']})")
    if result.get("abstained"):
        print(f"\n⚠ ABSTAINED\n  {result['answer']}")
        return
    print(f"\nANSWER :\n{result['answer']}")
    if result["citations"]:
        print(f"\nCITATIONS ({len(result['citations'])}):")
        for c in result["citations"]:
            print(f"  [{c.get('year','?')}] {c.get('title','?')[:55]}")