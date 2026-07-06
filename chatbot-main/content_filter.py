import logging
import re
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _build_hard_patterns() -> list[tuple[re.Pattern, str]]:
    patterns: list[tuple[re.Pattern, str]] = []

    def _p(pats: list[str], reason: str, flags: int = re.I | re.UNICODE) -> None:
        combined = "|".join(f"(?:{p})" for p in pats)
        patterns.append((re.compile(combined, flags), reason))

    _p([
        r"\bself[\s\-]?harm\b", r"\bcut(ting)?\s+(myself|yourself)\b",
        r"\bsuicid", r"\bkill\s+my(self)?\b",
        r"\bwant\s+to\s+die\b", r"\bend\s+(my|this)\s+life\b",
        r"إيذاء\s+النفس", r"أؤذي\s+نفسي", r"أنهي\s+(حياتي|حياتى)",
        r"أقتل\s+نفسي", r"الانتحار", r"أجرح\s+نفسي",
    ], reason="self_harm")

    _p([
        r"\b(cocaine|heroin|meth|crack|fentanyl|lsd|mdma|ecstasy)\b",
        r"\b(كوكايين|هيروين|ميث|فنتانيل|حشيش)\b",
        r"\billegal\s+drug",
        r"مخدرات\s+(علاج|دواء)",
    ], reason="drugs")

    _p([
        r"\b(sex|porn|nude|naked|masturbat)\b",
        r"جنس|إباحي|عارية|تعري",
    ], reason="sexual")

    return patterns


def _build_scope_patterns() -> list[re.Pattern]:
    patterns: list[re.Pattern] = []

    def _sp(pats: list[str]) -> None:
        combined = "|".join(f"(?:{p})" for p in pats)
        patterns.append(re.compile(combined, re.I | re.UNICODE))

    _sp([
        r"\b(politic|election|president|war|army)\b",
        r"(سياسة|انتخابات|حكومة|حرب|جيش)",
        r"\b(stock|crypto|bitcoin|forex|invest)\b",
        r"(بورصة|كريبتو|استثمار|فوركس)",
        r"\b(hack|malware|virus|phishing)\b",
        r"(اختراق|فيروس|هاكر)",
    ])

    return patterns


_HARD_PATTERNS  = _build_hard_patterns()
_SCOPE_PATTERNS = _build_scope_patterns()


_CLASSIFIER_SYSTEM = """\
You are a content moderation classifier for a medical diagnosis chatbot.

Classify the user message into ONE of these categories:
  - "safe"       : any medical question, symptoms, health concerns, general questions
  - "self_harm"  : self-harm, suicide, hurting oneself
  - "drugs"      : illegal recreational drugs
  - "sexual"     : sexual content
  - "off_topic"  : completely unrelated (politics, finance, hacking, etc.)

Rules:
- ANY medical question (even serious diseases like cancer, mental health, etc.) → "safe"
- Symptoms, medications, treatments, medical advice → "safe"
- General greetings or health-related chit-chat → "safe"
- If unsure → "safe"

Respond with ONLY the category word. Nothing else."""

_VALID_LABELS = {"safe", "self_harm", "drugs", "sexual", "off_topic"}


async def _llm_classify(text: str) -> str:
    from google.genai import types
    from key_manager import with_key_retry_async

    async def _call(client):
        return await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{_CLASSIFIER_SYSTEM}\n\nMessage: {text[:800]}",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=10,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        response, _key = await with_key_retry_async(_call)
        label = (response.text or "").strip().lower()
        return label if label in _VALID_LABELS else "safe"
    except Exception as e:
        logger.warning("Classifier error (all keys failed): %s — failing open (safe)", e)
        return "safe"


_BLOCK_MSGS: dict[str, dict[str, str]] = {
    "self_harm": {
        "ar": "⚠️ لاحظت إنك بتمر بوقت صعب. أنا مش متخصص في الدعم النفسي، لكن أرجوك تتواصل مع متخصص أو خط نجدة نفسية. أنت مهم. 🙏",
        "en": "⚠️ It sounds like you're going through something difficult. Please reach out to a mental health professional or crisis line. You matter. 🙏",
    },
    "drugs": {
        "ar": "⚠️ الموضوع ده خارج نطاق تخصصي. أنا هنا عشان مساعدتك طبياً! 🏥",
        "en": "⚠️ This topic is outside my scope. I'm here to help with medical concerns! 🏥",
    },
    "sexual": {
        "ar": "⚠️ المحتوى ده مش مناسب هنا. أنا متخصص في الاستشارات الطبية فقط. 🏥",
        "en": "⚠️ This content isn't appropriate here. I specialise in medical consultations only. 🏥",
    },
    "off_topic": {
        "ar": "أنا متخصص في الاستشارات الطبية فقط. لو عندك أي سؤال صحي، أنا هنا! 🏥",
        "en": "I specialise in medical consultations only. Ask me anything health-related! 🏥",
    },
}


def get_block_message(reason: str, lang: str) -> str:
    msgs = _BLOCK_MSGS.get(reason, _BLOCK_MSGS["off_topic"])
    return msgs.get(lang, msgs["en"])


@dataclass
class FilterResult:
    blocked: bool
    reason:  str  = ""
    is_hard: bool = False
    message: str  = ""


async def filter_message(
    text:    str,
    lang:    str,
    use_llm: bool = True,
) -> FilterResult:
    t = text.strip()

    for pattern, reason in _HARD_PATTERNS:
        if pattern.search(t):
            logger.warning("Hard block [%s]: %.60s", reason, t)
            msg = get_block_message(reason, lang)
            return FilterResult(blocked=True, reason=reason,
                                is_hard=True, message=msg)

    scope_hit = any(p.search(t) for p in _SCOPE_PATTERNS)

    if scope_hit:
        if use_llm:
            label = await _llm_classify(t)
            if label != "safe":
                logger.info("LLM block [%s]: %.60s", label, t)
                msg = get_block_message(label, lang)
                return FilterResult(blocked=True, reason=label,
                                    is_hard=False, message=msg)
        else:
            logger.warning(
                "Scope hit but LLM disabled — passing through: %.60s", t
            )

    return FilterResult(blocked=False)
