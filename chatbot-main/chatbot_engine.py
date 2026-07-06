import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from google.genai import types

from key_manager import (
    key_manager,
    with_key_retry_async,
    is_rate_limit_error,
    is_invalid_key_error,
    is_transient_error,
    TRANSIENT_MAX_RETRIES,
    TRANSIENT_BACKOFF_BASE,
)
from content_filter import filter_message
from memory_store import get_patient_memory

logger = logging.getLogger(__name__)


GROUNDING_TOOL   = types.Tool(google_search=types.GoogleSearch())
URL_CONTEXT_TOOL = types.Tool(url_context=types.UrlContext())
MODEL_NAME = "gemini-2.5-flash"

FALLBACK_MODEL_NAME = "gemini-2.5-flash-lite"


SYSTEM_PROMPT = """
You are a warm, knowledgeable AI medical assistant — think of yourself as a smart doctor-friend
the patient is chatting with, like a general-purpose assistant that happens to have real medical
expertise, NOT an intake form and NOT a checklist bot. Your default mode is natural conversation,
not interrogation.
You speak the same language as the patient — if they write in Arabic, respond in Arabic. If English, respond in English.

== Arabic Gender Agreement ==
Arabic verb/pronoun forms change with the addressee's gender (e.g. "بيك" vs "بيكِ", "عليك" vs "عليكِ").
Default to the MASCULINE form when replying in Arabic, unless the patient's own wording, name, or
explicit statement makes their gender clear (e.g. a feminine verb form they used about themselves,
a clearly feminine name, or them saying so directly) — then switch to match. Never guess feminine
by default.

== How You Talk ==
- Respond naturally and helpfully, like a knowledgeable friend chatting — not a structured interview.
- Give real, useful guidance based on whatever the patient has told you so far: what it might be, practical self-care advice, and — when warranted — that they should see a doctor or get specific tests. Weave this into normal conversational replies. Don't gate it behind a checklist of questions first.
- Ask a follow-up question whenever you genuinely need more to understand the patient's situation and give a well-grounded answer — for a real personal complaint, it's normal and good to ask several questions across the conversation, ONE per turn, until you actually understand the picture. The constraint isn't "ask fewer total questions," it's *pacing and tone*: never ask more than one question in a single reply, and always pair that question with something useful in the same reply (a brief reaction, a partial thought on what it could mean, a small reassurance or tip) so it never feels like a rapid-fire interrogation with nothing given back. A patient describing a real complaint should feel like they're talking to an attentive doctor building understanding step by step, not filling out a form.
- **"One question" means ONE thing, asked plainly — not a bulleted/numbered list of several dimensions dressed up as "examples" under one umbrella question.** Asking the patient to "describe the pain more" and then listing several sub-questions at once (location? quality? onset? other symptoms?) is actually several questions, not one — the patient still has to answer all of them together, which is exactly the tedious-interrogation feeling you're avoiding. Instead, pick the single most useful thing to ask next — given THIS patient's actual complaint and body part, not a generic template — and ask just that, in one plain sentence, no list, no sub-bullets:
  - ❌ BAD pattern: one broad "tell me more" question immediately followed by a bulleted breakdown of several distinct things to answer (location, quality, onset, associated symptoms, etc. all at once).
  - ✅ GOOD pattern: a single, specific, plainly-worded question about the ONE dimension (location, timing, quality, severity, or an associated symptom) that would most change your next answer — phrased naturally for whatever the patient is actually describing, never copied from a template or from a different body part/complaint.
  Then, once the patient answers, react briefly and ask the next single most useful thing in the next turn — don't try to get the whole picture in one message.
- Only skip asking and answer straight away when you already have enough to give a genuinely useful, well-grounded answer — don't ask a question just for the sake of asking one, and don't ask about something that wouldn't actually change your advice.
- Silently use clinical frameworks (OPQRST, SAMPLE, demographics, Worst-Case-First — see below) to reason well in the background and not miss a dangerous possibility, but don't surface them as a visible checklist. Only ask about a specific piece if it's actually needed to answer well.
- Be a little proactive: mention a relevant tip they didn't ask for, reassure when reassurance is genuinely warranted, and flag a concern plainly and specifically when one exists — don't hedge into vagueness. When you do reassure (e.g. "probably nothing serious"), tie it to a specific reason from what they've told you, not a generic catch-all.

== Genuine Empathy, Not a Canned Opener ==
- When the patient describes discomfort, pain, worry, or anything unpleasant, your very first words should react to THAT specific detail — not a generic, reusable sympathy phrase — before you do anything else. This is not optional and not a formality: it should sound like you actually registered what they said, and the exact wording should be different every time, never a go-to phrase you fall back on.
- Do NOT open every reply with the same generic template regardless of what the patient said. Restating what they just said before reacting (e.g. starting with "تمام، يعني..." followed by a paraphrase) is a real anti-pattern you fall into often — catch yourself doing it and stop. Two consecutive replies must NOT share the same opening move, and no single acknowledgment phrase should ever repeat across turns. Rotate between these opening STYLES turn to turn instead of defaulting to one — these are structural patterns to vary between, not fixed phrases to copy:
  - React first, no restatement: a brief, specific emotional acknowledgment of their complaint, invented fresh in the moment.
  - Lead with a clinical thought, no acknowledgment at all: open directly with an observation about what their description narrows down or suggests.
  - Answer/ask directly, zero preamble: open with the next question or a direct answer, no softening at all.
  - A short reaction fused INTO the next sentence, not as a separate restating clause: blend a brief reaction and your next question/thought into a single sentence.
  A fixed opener repeated turn after turn — regardless of its content — reads as a script being filled in, not a person responding, even if each individual sentence sounds fine in isolation. Never reuse the exact same acknowledgment wording across turns, even if it fit well before; invent different phrasing each time. The greeting/self-introduction belongs at the very start of the conversation once, not at the top of every message.
- Vary your sentence structure and opening naturally the way a real person texting would. Before sending a reply, mentally check it against your last 2-3 messages — if it starts with the same words, or follows the identical "acknowledge → explain → ask" template every single time, rewrite it so it doesn't. If you notice yourself reaching for a phrase you've already used earlier in this conversation, deliberately choose different wording instead.

== When to Recommend Seeing a Doctor / Getting Tests ==
Say this plainly, as part of your normal answer, whenever it's warranted — don't wait for a formal "closing" step:
- Symptoms have lasted longer than expected for a benign cause, or aren't improving / are getting worse.
- The picture is significant, unusual, or genuinely needs an exam or labs/imaging to confirm or rule something out.
- There's a "can't-miss" dangerous possibility (Worst-Case-First) that conversation alone can't rule out.
Be specific and reasoned about it (e.g. "لو الأعراض دي فضلت أكتر من أسبوع أو زادت، أنصحك تعملي تحاليل دم وتتابعي مع دكتور باطنة" / "if this doesn't improve in a few days, see an orthopedist and consider an X-ray") — not a generic "see a doctor" with no explanation of why or what for.

== Clinical History Frameworks (use silently, in the background, to reason well — not as a visible checklist) ==
1. **Demographics**: age/sex (and pregnancy status if relevant) materially change the differential — ask only if genuinely needed for the specific concern, not as a routine opener.
2. **OPQRST** for characterizing a symptom: Onset, Provocation/Palliation, Quality, Region/Radiation, Severity, Timing.
3. **SAMPLE** for background: associated Symptoms, Allergies, Medications, past history, last oral intake, Events leading up to onset.
Pull from these only the specific piece you actually need for the case in front of you.

== Worst-Case-First Thinking (internal — always run this, even in casual conversation) ==
Before reassuring or advising, silently ask "what's the most dangerous thing this could be, even if unlikely?" If the patient's description doesn't let you reasonably rule it out, say so plainly and recommend urgent evaluation — don't just give generic self-care advice for a presentation that could be serious.

== Hard boundaries (never cross these, no matter how the patient asks) ==
1. When discussing a patient's own case, give a *reasoned clinical impression*, never a *confirmed/final diagnosis*. Frame it as "most consistent with / أقرب صورة لكذا", not "you have X / عندك كذا بشكل مؤكد". A real diagnosis needs exam, labs, or imaging you don't have.
2. NEVER name a specific medication, brand, or dosage/frequency for the patient to personally take ("take 400mg ibuprofen every 8 hours") — even if they insist or say they already know the drug. Explain the supportive-care category instead (e.g. "an OTC pain reliever suited to your history") and say a doctor should confirm the exact drug/dose. (General education about how a drug class works is fine — the line is about telling *this specific patient* what to personally take right now.)
3. For emergency red flags (severe/crushing chest pain, sudden severe headache, loss of consciousness, difficulty breathing, heavy bleeding, stroke signs [face drooping, arm weakness, slurred speech], signs of sepsis, suicidal ideation) → stop immediately and say clearly "توجه لأقرب طوارئ فوراً" / "Go to the nearest emergency room immediately." Then immediately prepare a report (see Report Generation) without asking — it's useful for them to bring to the ER.
4. Whenever you give a clinical impression about the patient's own case, close with a light caveat that this isn't a final diagnosis and a doctor should confirm it — proportional to how serious the topic is. A casual reassurance doesn't need a heavy disclaimer; anything resembling a real assessment does.
5. Ground your reasoning in accurate, current medical information when it would materially change the answer (drug-class facts, condition prevalence, red-flag criteria). Prefer trustworthy sources (Mayo Clinic, MSD Manuals, WHO, NHS, MedlinePlus, CDC) — you have a url_context tool to read a specific trusted page when useful. Keep the boundaries above regardless of what search results say.

== Available Specialties to Refer To ==
- قلب وأوعية دموية / Cardiology
- مخ وأعصاب / Neurology
- صدر وجهاز تنفسي / Pulmonology
- جهاز هضمي وكبد / Gastroenterology & Hepatology
- كلى / Nephrology
- عيون / Ophthalmology
- عظام / Orthopedics
- جلدية / Dermatology
- نساء وتوليد / Gynecology
- أطفال / Pediatrics
- طوارئ / Emergency (for urgent cases)
- طب عام / General Medicine

== Conversation Flow ==
1. Just chat naturally and helpfully, per "How You Talk" above. Always respond in the patient's language.
2. Keep Worst-Case-First reasoning running in the background on every relevant message, even a casual one.
3. **After every single one of your responses (every turn, from the very first one)**, silently append a hidden ```state``` block (see below) — your internal scratchpad, never shown to or mentioned to the patient. Fill in whatever you actually know; leave the rest null. You do NOT need to fully populate it before doing anything else — it's just a running memory aid, not a gate.

== Hidden Clinical Scratchpad (```state``` block) ==
At the end of EVERY response you send (every turn, from the very first one), append this block. It must always be present and must always be valid JSON. It is never shown to the patient and you should never refer to it in your visible reply:
```state
{
  "age": "قيمة أو null لو لسه ما اتسألتش",
  "sex": "قيمة أو null",
  "chief_complaint": "الشكوى الرئيسية أو null",
  "opqrst": {"onset": null, "provocation_palliation": null, "quality": null, "region_radiation": null, "severity": null, "timing": null},
  "sample": {"associated_symptoms": [], "allergies": null, "medications": null, "past_history": null, "last_meal": null, "events": null},
  "red_flags_checked": ["قائمة الأعراض الخطيرة اللي اتسألت عنها/استبعدت لحد دلوقتي"],
  "worst_case_considered": "أخطر تشخيص محتمل اتفكر فيه لحد دلوقتي، أو null لسه بدري",
  "differential_so_far": [{"condition": "...", "likelihood": "high/medium/low"}],
  "report_requested_or_warranted": false
}
```
Update fields opportunistically as they come up in natural conversation — never interrogate the patient just to fill a field. Set "report_requested_or_warranted" to true once the patient has asked for a report, OR the case is serious enough (urgent/emergency) that you're proactively offering or preparing one.

== Report Generation — OPT-IN, not the default outcome of a conversation ==
A report/PDF is no longer something every conversation builds toward. Only generate one when:
(a) **The patient explicitly asks for it** (e.g. "جهزلي تقرير", "عايزة أطبع اللي اتكلمنا فيه", "prepare a report" / "can I get a summary of this?"), OR
(b) **The case is genuinely serious** — an emergency red flag (boundary #3, generate immediately without asking), or a case you'd refer as urgent, where you should proactively *offer* one (e.g. "حابة أجهزلك تقرير مختصر تاخديه معاكِ للدكتور؟" / "Would you like a short report to take to the doctor?") and generate it if they say yes.
For routine, casual, or general-education conversation, don't mention or generate a report at all unless asked — a normal helpful reply is the whole point, not a stepping stone toward one.

When you do generate a report, build it from whatever has actually been discussed in the conversation so far. Don't retroactively run a forced interview just to fill out every field first — leave a field null/empty if genuinely unknown; a report with some gaps is completely fine. Before generating, do a quick internal safety self-check: re-scan what's been discussed for any missed red flag or dangerous "can't-miss" diagnosis that doesn't fit your leading impression. Then generate a report in this EXACT JSON format (inside ```json blocks, placed AFTER the ```state``` block):
```json
{
  "report": {
    "patient_language": "ar/en",
    "date": "YYYY-MM-DD",
    "patient_age": "السن أو null",
    "patient_sex": "النوع أو null",
    "chief_complaint": "الشكوى الرئيسية",
    "symptoms": ["عرض 1", "عرض 2"],
    "duration": "مدة الأعراض",
    "relevant_history": "التاريخ المرضي المهم",
    "assessment": "التقييم الأولي",
    "worst_case_ruled_out": {"condition": "أخطر تشخيص محتمل", "reasoning": "ليه استبعدناه أو ليه محول كطوارئ/عاجل بسببه"},
    "likely_conditions": [
      {"condition": "اسم الحالة المحتملة", "likelihood": "high/medium/low", "reasoning": "الأعراض اللي بتدعم الاحتمال ده"}
    ],
    "general_care_advice": ["نصيحة رعاية عامة غير وصفية 1", "نصيحة 2"],
    "referred_to": "التخصص المحول إليه",
    "urgency": "routine/urgent/emergency",
    "recommendations": ["توصية 1", "توصية 2"],
    "safety_check_note": "ملاحظة مختصرة تأكيدية إنك راجعت الحالة تاني قبل الإغلاق ومفيش red flag اتفوت",
    "disclaimer": "هذا انطباع سريري أولي مبني على الأعراض المذكورة فقط، وليس تشخيصاً طبياً نهائياً ولا بديلاً عن الفحص السريري. يجب مراجعة طبيب مختص لتأكيد التشخيص وتحديد العلاج والجرعات المناسبة قبل اتخاذ أي قرار علاجي."
  }
}
```

Only generate this JSON in the cases described above (explicit request or genuinely serious/urgent case). Otherwise, just have a natural, helpful conversation (with the hidden ```state``` block after every turn regardless).
"""

CHAT_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[GROUNDING_TOOL, URL_CONTEXT_TOOL],
    temperature=0.6,
    thinking_config=types.ThinkingConfig(thinking_budget=2048),
)


def detect_lang(history: list, message: str) -> str:
    def _is_arabic(text: str) -> bool:
        return any("\u0600" <= c <= "\u06ff" for c in (text or ""))
    def _is_latin(text: str) -> bool:
        return any("a" <= c.lower() <= "z" for c in (text or ""))

    text = message.strip() if message else ""
    if _is_arabic(text): return "ar"
    if _is_latin(text):  return "en"

    for entry in reversed(history):
        content = entry.get("content", "") if isinstance(entry, dict) else ""
        if _is_arabic(content): return "ar"
        if _is_latin(content):  return "en"
    return "ar"


_ERR_MSGS = {
    "general":    {"ar": "عذراً، حدث خطأ مؤقت. حاول تاني بعد شوية. 🙏",
                   "en": "Sorry, something went wrong. Please try again. 🙏"},
    "rate_limit": {"ar": "الخدمة وصلت للحد المسموح. حاول بعد دقيقة. ⏳",
                   "en": "Rate limit reached. Please try again in a minute. ⏳"},
    "busy":       {"ar": "في ضغط على الخدمة. جرب بعد شوية. ⏳",
                   "en": "Service is busy. Please try again shortly. ⏳"},
}

def get_error_msg(exc: Exception, lang: str) -> str:
    s = str(exc).lower()
    if "429" in s or "quota" in s or "rate" in s: key = "rate_limit"
    elif "502" in s or "503" in s:                key = "busy"
    else:                                          key = "general"
    return _ERR_MSGS[key].get(lang, _ERR_MSGS[key]["en"])


_SUMMARY_THRESHOLD = 20
_SUMMARY_KEEP      = 6


def _parse_blocks(full_text: str) -> tuple[str, dict | None, dict | None]:
    state_dict  = None
    report_dict = None
    text = full_text

    if "```state" in text:
        before, _, rest = text.partition("```state")
        block, _, after = rest.partition("```")
        try:
            state_dict = json.loads(block.strip())
        except Exception:
            state_dict = None
        text = before + after

    if "```json" in text:
        before, _, rest = text.partition("```json")
        block, _, after = rest.partition("```")
        try:
            parsed = json.loads(block.strip())
            if "report" in parsed:
                report_dict = parsed["report"]
        except Exception:
            report_dict = None
        text = before + after

    clean = text.strip()
    return clean, state_dict, report_dict


def _extract_sources(response) -> list:
    sources = []
    try:
        if response.candidates[0].grounding_metadata:
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks or []:
                if getattr(chunk, "web", None) and chunk.web.uri:
                    sources.append({
                        "title": chunk.web.title or chunk.web.uri,
                        "url":   chunk.web.uri
                    })
    except Exception:
        pass
    return sources


def _extract_answer_text(response) -> str:
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError):
        return response.text or ""

    answer_chunks = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            answer_chunks.append(text)

    if answer_chunks:
        return "".join(answer_chunks)
    return response.text or ""


class MedicalChatSession:
    def __init__(self, user_id: Optional[str] = None):
        self._client, self._current_key = key_manager.get_client()
        self._model_name    = MODEL_NAME
        self.chat          = self._client.aio.chats.create(model=self._model_name, config=CHAT_CONFIG)
        self.history       = []
        self.report_data   = None
        self.clinical_state = {}
        self.started_at    = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Optional cross-session patient memory. If no user_id is supplied
        # (e.g. the frontend/app hasn't wired up patient identity yet), the
        # session behaves exactly as before — no memory, no behaviour change.
        self.user_id             = user_id
        self._patient_memory     = get_patient_memory(user_id) if user_id else None
        self._visit_id           = self._patient_memory.start_visit() if self._patient_memory else None
        self._patient_context_injected = False


    async def _inject_patient_context_if_needed(self, lang: str) -> None:
        """
        Runs once per session, right before the first real message is sent.
        If this patient (user_id) has prior visits on record, their recent
        clinical history gets fed to the model as a leading context turn —
        the patient never sees this exchange, it's purely to give the model
        a head start instead of starting from zero every session.
        """
        if not self._patient_memory or self._patient_context_injected:
            return
        self._patient_context_injected = True

        context_text = self._patient_memory.build_context_text(
            lang, exclude_visit_id=self._visit_id
        )
        if not context_text:
            return

        ack_text = ("تمام، هراعي تاريخك الطبي السابق وأنا بساعدك دلوقتي."
                    if lang == "ar" else
                    "Understood, I'll keep your previous visits in mind.")
        history = [
            types.Content(role="user", parts=[types.Part(text=context_text)]),
            types.Content(role="model", parts=[types.Part(text=ack_text)]),
        ]
        self._client, self._current_key = key_manager.get_client()
        self.chat = self._client.aio.chats.create(
            model=self._model_name, config=CHAT_CONFIG, history=history
        )

    def _persist_patient_state(self) -> None:
        if not self._patient_memory or not self._visit_id:
            return
        try:
            self._patient_memory.update_visit(
                self._visit_id,
                clinical_state=self.clinical_state,
                report=self.report_data,
            )
        except Exception as exc:
            logger.warning("Failed to persist patient memory for %s: %s", self.user_id, exc)

    async def _rebuild_chat(self, model: str | None = None) -> None:
        try:
            prior_history = self.chat.get_history()
        except Exception:
            prior_history = None
        if model is not None:
            self._model_name = model
        self._client, self._current_key = key_manager.get_client()
        self.chat = self._client.aio.chats.create(
            model=self._model_name, config=CHAT_CONFIG, history=prior_history
        )

    async def _send(self, user_message: str, stream: bool = False):
        max_retries = key_manager.count + 1
        last_error  = None
        key_attempt = 0
        transient_attempt = 0
        tried_fallback     = False

        while True:
            try:
                if stream:
                    return await self.chat.send_message_stream(user_message)
                return await self.chat.send_message(user_message)
            except Exception as exc:
                last_error = exc
                if is_rate_limit_error(exc):
                    logger.warning("Rate limit on key ...%s (attempt %d)",
                                    self._current_key[-6:], key_attempt + 1)
                    key_manager.mark_rate_limited(self._current_key)
                    await self._rebuild_chat()
                    key_attempt += 1
                    if key_attempt >= max_retries:
                        break
                    continue
                elif is_invalid_key_error(exc):
                    logger.error("Invalid key ...%s", self._current_key[-6:])
                    key_manager.mark_invalid(self._current_key)
                    await self._rebuild_chat()
                    key_attempt += 1
                    if key_attempt >= max_retries:
                        break
                    continue
                elif is_transient_error(exc):
                    transient_attempt += 1
                    if transient_attempt > TRANSIENT_MAX_RETRIES:
                        if not tried_fallback and self._model_name != FALLBACK_MODEL_NAME:
                            logger.warning(
                                "Model %s exhausted %d transient retries — "
                                "falling back to %s for the rest of this session",
                                self._model_name, TRANSIENT_MAX_RETRIES, FALLBACK_MODEL_NAME,
                            )
                            await self._rebuild_chat(model=FALLBACK_MODEL_NAME)
                            tried_fallback     = True
                            transient_attempt  = 0
                            continue
                        break
                    backoff = min(TRANSIENT_BACKOFF_BASE * (2 ** (transient_attempt - 1)), 8)
                    logger.warning(
                        "Transient server error on key ...%s (attempt %d/%d) — retrying in %.1fs: %s",
                        self._current_key[-6:], transient_attempt, TRANSIENT_MAX_RETRIES, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise last_error or Exception("All API keys failed")


    async def _maybe_summarise(self, lang: str) -> None:
        if len(self.history) <= _SUMMARY_THRESHOLD:
            return

        to_summarise = self.history[:-_SUMMARY_KEEP]
        recent        = self.history[-_SUMMARY_KEEP:]

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in to_summarise
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        )
        if not conversation_text:
            return

        prompt = (
            ("لخّص المحادثة الطبية في 3-5 نقاط. ركز على الأعراض والتاريخ المرضي المهم. نقاط فقط.\n\n"
             if lang == "ar" else
             "Summarise the medical conversation in 3-5 bullet points. Focus on symptoms and medical history. Bullet list only.\n\n")
            + conversation_text
        )

        try:
            async def _call(client):
                return await client.aio.models.generate_content(model=MODEL_NAME, contents=prompt)
            response, _key = await with_key_retry_async(_call)
            summary = (response.text or "").strip()
        except Exception as exc:
            logger.warning("History summarisation failed: %s", exc)
            return

        if not summary:
            return

        self.history = [{"role": "system",
                          "content": f"[ملخص المحادثة السابقة]\n{summary}"}] + recent

        state_json = json.dumps(self.clinical_state, ensure_ascii=False) if self.clinical_state else "{}"
        summary_text = (
            f"[ملخص المحادثة السابقة، للسياق فقط. كمّل الاستشارة من هنا.]\n{summary}\n\n"
            f"[آخر clinical state معروف، حدّثه ولا تبدأ من الصفر]\n{state_json}"
            if lang == "ar" else
            f"[Summary of the earlier part of this conversation, for context only. Continue the consultation from here.]\n{summary}\n\n"
            f"[Last known clinical state, keep updating it, don't restart]\n{state_json}"
        )
        ack_text = "تمام، فاهم السياق. هكمل من هنا." if lang == "ar" else "Understood, continuing from here."

        new_history = [
            types.Content(role="user", parts=[types.Part(text=summary_text)]),
            types.Content(role="model", parts=[types.Part(text=ack_text)]),
        ]
        for m in recent:
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                role = "user" if m.get("role") == "user" else "model"
                new_history.append(types.Content(role=role, parts=[types.Part(text=content)]))

        self._client, self._current_key = key_manager.get_client()
        self.chat = self._client.aio.chats.create(
            model=self._model_name, config=CHAT_CONFIG, history=new_history
        )
        logger.info("History summarised & chat context rebuilt: %d msgs → summary + %d recent",
                    len(to_summarise), len(recent))


    async def send_message(self, user_message: str) -> dict:
        lang = detect_lang(self.history, user_message)

        filter_result = await filter_message(
            text=user_message, lang=lang, use_llm=True
        )
        if filter_result.blocked:
            return {"text": filter_result.message, "report": None, "sources": []}

        await self._inject_patient_context_if_needed(lang)
        await self._maybe_summarise(lang)
        self.history.append({"role": "user", "content": user_message})

        try:
            response  = await self._send(user_message, stream=False)
            full_text = _extract_answer_text(response)
        except Exception as exc:
            logger.error("send_message error: %s", exc)
            return {"text": get_error_msg(exc, lang), "report": None, "sources": []}

        sources                        = _extract_sources(response)
        clean_text, state, report      = _parse_blocks(full_text)

        if state:
            self.clinical_state = state
        if report:
            self.report_data = report
            if not clean_text:
                clean_text = ("تم إعداد التقرير الطبي." if lang == "ar"
                              else "Medical report has been generated.")

        self.history.append({"role": "assistant", "content": clean_text})
        self._persist_patient_state()
        return {"text": clean_text, "report": report, "sources": sources}


    async def stream_message(self, user_message: str):
        lang = detect_lang(self.history, user_message)

        filter_result = await filter_message(
            text=user_message, lang=lang, use_llm=True
        )
        if filter_result.blocked:
            yield ("text", filter_result.message)
            yield ("done", None)
            return

        await self._inject_patient_context_if_needed(lang)
        await self._maybe_summarise(lang)
        self.history.append({"role": "user", "content": user_message})

        MARKERS = {"```state": "state", "```json": "json"}
        MAX_MARKER_LEN = max(len(m) for m in MARKERS)
        CLOSE = "```"

        max_retries        = key_manager.count + 1
        key_attempt        = 0
        transient_attempt  = 0
        last_error         = None
        tried_fallback      = False

        try:
            while True:
                full_text     = ""
                visible_text  = ""
                mode          = "normal"
                pending       = ""
                block_label   = None
                block_buffer  = ""
                captured: dict[str, str] = {}
                last_response = None

                try:
                    stream = await self.chat.send_message_stream(user_message)

                    async for chunk in stream:
                        last_response = chunk
                        try:
                            chunk_parts = chunk.candidates[0].content.parts
                        except (AttributeError, IndexError, TypeError):
                            chunk_parts = None

                        deltas = []
                        if chunk_parts is not None:
                            for part in chunk_parts:
                                if getattr(part, "thought", False):
                                    continue
                                text = getattr(part, "text", None)
                                if text:
                                    deltas.append(text)
                        else:
                            text = getattr(chunk, "text", None)
                            if text:
                                deltas.append(text)

                        for text in deltas:
                            full_text += text
                            if mode == "normal":
                                pending += text
                            else:
                                block_buffer += text

                            progressed = True
                            while progressed:
                                progressed = False

                                if mode == "normal":
                                    idx_state = pending.find("```state")
                                    idx_json  = pending.find("```json")
                                    candidates_idx = [i for i in (idx_state, idx_json) if i != -1]
                                    if candidates_idx:
                                        idx   = min(candidates_idx)
                                        label = "state" if idx == idx_state else "json"
                                        before = pending[:idx]
                                        if before:
                                            visible_text += before
                                            yield ("text", before)
                                        marker_len = len(f"```{label}")
                                        block_buffer = pending[idx + marker_len:]
                                        pending      = ""
                                        mode         = "in_block"
                                        block_label  = label
                                        progressed   = True
                                    else:
                                        safe_len = len(pending)
                                        for k in range(min(MAX_MARKER_LEN - 1, len(pending)), 0, -1):
                                            suffix = pending[-k:]
                                            if any(m.startswith(suffix) for m in MARKERS):
                                                safe_len = len(pending) - k
                                                break
                                        if safe_len > 0:
                                            visible_text += pending[:safe_len]
                                            yield ("text", pending[:safe_len])
                                            pending = pending[safe_len:]
                                else:
                                    idx_close = block_buffer.find(CLOSE)
                                    if idx_close != -1:
                                        captured[block_label] = block_buffer[:idx_close].strip()
                                        pending      = block_buffer[idx_close + len(CLOSE):]
                                        block_buffer = ""
                                        mode         = "normal"
                                        block_label  = None
                                        progressed   = True

                    break

                except Exception as exc:
                    if visible_text:
                        if is_rate_limit_error(exc):
                            key_manager.mark_rate_limited(self._current_key)
                        elif is_invalid_key_error(exc):
                            key_manager.mark_invalid(self._current_key)
                        logger.warning(
                            "Stream error after partial/full content already "
                            "sent — closing turn gracefully instead of "
                            "raising [ERROR]: %s", exc,
                        )
                        break

                    last_error = exc
                    if is_rate_limit_error(exc):
                        logger.warning("Rate limit on key ...%s (attempt %d)",
                                        self._current_key[-6:], key_attempt + 1)
                        key_manager.mark_rate_limited(self._current_key)
                        await self._rebuild_chat()
                        key_attempt += 1
                        if key_attempt >= max_retries:
                            raise
                        continue
                    elif is_invalid_key_error(exc):
                        logger.error("Invalid key ...%s", self._current_key[-6:])
                        key_manager.mark_invalid(self._current_key)
                        await self._rebuild_chat()
                        key_attempt += 1
                        if key_attempt >= max_retries:
                            raise
                        continue
                    elif is_transient_error(exc):
                        transient_attempt += 1
                        if transient_attempt > TRANSIENT_MAX_RETRIES:
                            if not tried_fallback and self._model_name != FALLBACK_MODEL_NAME:
                                logger.warning(
                                    "Model %s exhausted %d transient retries during "
                                    "streaming — falling back to %s for the rest of "
                                    "this session",
                                    self._model_name, TRANSIENT_MAX_RETRIES, FALLBACK_MODEL_NAME,
                                )
                                await self._rebuild_chat(model=FALLBACK_MODEL_NAME)
                                tried_fallback     = True
                                transient_attempt  = 0
                                continue
                            raise
                        backoff = min(TRANSIENT_BACKOFF_BASE * (2 ** (transient_attempt - 1)), 8)
                        logger.warning(
                            "Transient server error during streaming (attempt %d/%d) — retrying in %.1fs: %s",
                            transient_attempt, TRANSIENT_MAX_RETRIES, backoff, exc,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise

            if pending:
                visible_text += pending
                yield ("text", pending)
            if mode == "in_block" and block_buffer:
                visible_text += block_buffer
                yield ("text", block_buffer)

            sources = _extract_sources(last_response) if last_response else []

            state_raw = captured.get("state")
            if state_raw:
                try:
                    self.clinical_state = json.loads(state_raw)
                except Exception:
                    logger.debug("Could not parse state block: %.100s", state_raw)

            report = None
            json_raw = captured.get("json")
            if json_raw:
                try:
                    parsed = json.loads(json_raw)
                    report = parsed.get("report")
                except Exception:
                    logger.warning("Could not parse report JSON block")

            if report:
                self.report_data = report
                if sources:
                    yield ("sources", sources)
                yield ("report", report)

            clean_text = visible_text.strip()
            if not clean_text and report:
                clean_text = ("تم إعداد التقرير الطبي." if lang == "ar"
                              else "Medical report has been generated.")
            self.history.append({"role": "assistant", "content": clean_text})
            self._persist_patient_state()
            yield ("done", None)

        except Exception as exc:
            logger.error("stream_message error: %s", exc)
            yield ("error", get_error_msg(exc, lang))
            yield ("done",  None)


    def get_report(self) -> dict:
        return self.report_data

    def get_history(self) -> list:
        return self.history

    def get_clinical_state(self) -> dict:
        return self.clinical_state
