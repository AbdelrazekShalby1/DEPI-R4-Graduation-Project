from __future__ import annotations

import pickle
import asyncio
import os
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - the app still runs without CORS.
    CORS = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
# Model and chatbot folders are checked in directly under the repository root.
# Resolve from this file so launching from another working directory also works.
MODELS_DIR = PROJECT_DIR
ML_MODELS_DIR = MODELS_DIR / "ml-models"
CHATBOT_DIR = MODELS_DIR / "chatbot-main"
BRAIN_DETECTION_DIR = MODELS_DIR / "brain_tomor detection"
BRAIN_DETECTION_EXTRACTED_DIR = BRAIN_DETECTION_DIR / "last"

TEMP_DIR = Path(tempfile.gettempdir())
RUNTIME_DIR = BASE_DIR / ".runtime"
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR),
    static_url_path="/static",
)

if CORS:
    CORS(app)


PAGE_NAMES = {
    "index",
    "about",
    "ml",
    "cv",
    "nlp",
    "kidney",
    "liver",
    "lung",
    "alz",
    "epilepsy",
    "brain",
    "brain_classification",
    "bone",
    "chest",
    "covid",
}

CV_PAGES = {
    "brain_classification": {
        "title": "Brain Tumor Classification",
        "heading": "Brain Tumor Classification",
        "description": "Upload a Brain MRI image for AI-powered tumor classification",
        "upload_title": "Drag & Drop Brain MRI Image",
        "preview_alt": "Brain MRI Preview",
        "button_text": "Classify MRI",
        "guidelines_title": "Upload Guidelines",
        "guidelines": [
            "Upload a clear Brain MRI image.",
            "Supported formats: JPG, PNG, JPEG.",
            "Maximum file size: 10 MB.",
            "Use original MRI scans for best results.",
        ],
        "analysis_title": "AI Classification",
        "analysis_items": [
            "MobileNet-based classifier.",
            "Classifies MRI scans as tumor or no tumor.",
            "Fast prediction from uploaded images.",
        ],
        "about_title": "About Brain Tumor Classification",
        "about_text": "This model classifies Brain MRI images to help identify whether a tumor pattern is present.",
    },
    "brain": {
        "title": "Brain Tumor Detection",
        "heading": "Brain Tumor Detection",
        "description": "Upload a Brain MRI image for AI-powered detection",
        "upload_title": "Drag & Drop Brain MRI Image",
        "preview_alt": "Brain MRI Preview",
        "button_text": "Detect Tumor",
        "guidelines_title": "Upload Guidelines",
        "guidelines": [
            "Upload a clear Brain MRI image.",
            "Supported formats: JPG, PNG, JPEG.",
            "Maximum file size: 10 MB.",
            "Use original MRI scans for best results.",
        ],
        "analysis_title": "AI Detection",
        "analysis_items": [
            "Brain tumor detection endpoint.",
            "Uses the detection model file when available.",
            "Returns a clear setup error if the model artifact is incomplete.",
        ],
        "about_title": "About Brain Tumor Detection",
        "about_text": "This clinic is reserved for the separate Brain Tumor Detection model.",
    },
    "chest": {
        "title": "Chest X-Ray Classification",
        "heading": "Chest X-Ray Classification",
        "description": "Upload a Chest X-Ray image for AI-powered classification",
        "upload_title": "Drag & Drop Chest X-Ray",
        "preview_alt": "Chest X-Ray Preview",
        "button_text": "Classify X-Ray",
        "guidelines_title": "Upload Guidelines",
        "guidelines": [
            "Upload a clear Chest X-Ray image.",
            "Supported formats: JPG, PNG, JPEG.",
            "Maximum file size: 10 MB.",
            "Use original X-Ray images.",
        ],
        "analysis_title": "AI Classification",
        "analysis_items": [
            "Chest X-Ray classification model.",
            "Classifies images as normal or pneumonia pattern.",
            "Instant prediction from uploaded images.",
        ],
        "about_title": "About Chest X-Ray Classification",
        "about_text": "This model classifies chest X-Ray images using the available pneumonia classifier.",
    },
}


def normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = []
    for char in text:
        if char.isalnum():
            normalized.append(char)
        else:
            normalized.append("_")
    return "_".join("".join(normalized).split("_"))


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def to_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    lowered = text.lower()
    common_values = {
        "yes": 1.0,
        "no": 0.0,
        "true": 1.0,
        "false": 0.0,
        "male": 1.0,
        "female": 0.0,
        "current": 2.0,
        "former": 1.0,
        "never": 0.0,
        "good": 1.0,
        "poor": 0.0,
        "urban": 1.0,
        "rural": 0.0,
    }
    if lowered in common_values:
        return common_values[lowered]
    try:
        return float(text)
    except ValueError:
        return float(default)


def encoder_classes(encoder: Any) -> list[str]:
    classes = getattr(encoder, "classes_", [])
    return [str(item) for item in classes]


def encode_value(encoder: Any, value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)

    classes = encoder_classes(encoder)
    text = str(value).strip()
    if text in classes:
        return float(encoder.transform([text])[0])

    normalized_text = normalize_label(text)
    for cls in classes:
        if normalize_label(cls) == normalized_text:
            return float(encoder.transform([cls])[0])

    return float(default)


MODEL_ALIASES = {
    "kidney": {
        "age_of_the_patient": ["age"],
        "blood_pressure_(mm/hg)": ["blood_pressure", "blood_pressure_mm_hg"],
        "specific_gravity_of_urine": ["specific_gravity"],
        "albumin_in_urine": ["albumin"],
        "sugar_in_urine": ["sugar"],
        "random_blood_glucose_level_(mg/dl)": ["blood_glucose", "blood_glucose_mg_dl"],
        "serum_creatinine_(mg/dl)": ["serum_creatinine", "serum_creatinine_mg_dl"],
        "hemoglobin_level_(gms)": ["hemoglobin", "hemoglobin_g_dl"],
        "body_mass_index_(bmi)": ["bmi"],
    },
    "liver": {
        "Sym_Abdominal_Pain": ["abdominal_pain"],
        "Sym_Dark_Urine": ["dark_urine"],
        "Comorb_Genetic_History": ["genetic_history"],
    },
    "lung": {
        "smoking_years": ["smoking"],
        "air_pollution_index": ["air_pollution"],
        "alcohol_units_per_week": ["alcohol_use"],
        "family_history_cancer": ["genetic_risk"],
        "copd": ["chronic_lung_disease"],
    },
    "alzheimer": {
        "cognitive_test_score": ["mmse_score", "memory_test", "cognitive_score"],
        "family_history_of_alzheimer’s": ["family_history"],
        "education_level": ["education_years"],
        "physical_activity_level": ["physical_activity"],
    },
    "epileptic": {
        "x1": ["eeg_signal_value"],
        "x2": ["brain_activity_score"],
        "x3": ["frequency_level"],
        "x4": ["signal_amplitude"],
        "x5": ["seizure_history"],
        "x6": ["medication"],
        "x7": ["stress_level"],
    },
}


PREDICTION_LABELS = {
    "kidney": {0: "No Kidney Disease", 1: "Kidney Disease Detected"},
    "lung": {0: "Low Lung Cancer Risk", 1: "High Lung Cancer Risk"},
    "epileptic": {0: "No Seizure Detected", 1: "Seizure Detected"},
}

OUTPUT_LABELS = {
    "alzheimer": {
        "No": "No Alzheimer's Detected",
        "Yes": "Alzheimer's Risk Detected",
    },
}


class ModelLoadError(RuntimeError):
    pass


class MLModel:
    def __init__(
        self,
        key: str,
        model_path: Path,
        scaler_path: Path,
        *,
        feature_encoders_path: Path | None = None,
        target_encoder_path: Path | None = None,
    ) -> None:
        self.key = key
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_encoders_path = feature_encoders_path
        self.target_encoder_path = target_encoder_path
        self.model = None
        self.scaler = None
        self.feature_encoders = {}
        self.target_encoder = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            self.model = load_pickle(self.model_path)
            self.scaler = load_pickle(self.scaler_path)
            if self.feature_encoders_path and self.feature_encoders_path.exists():
                self.feature_encoders = load_pickle(self.feature_encoders_path)
            if self.target_encoder_path and self.target_encoder_path.exists():
                self.target_encoder = load_pickle(self.target_encoder_path)
        except ImportError as exc:
            raise ModelLoadError(
                f"Missing Python package while loading {self.key}: {exc.name}"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"Could not load {self.key} model: {exc}") from exc

    @property
    def feature_names(self) -> list[str]:
        names = getattr(self.scaler, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]
        feature_count = int(getattr(self.scaler, "n_features_in_", 0) or 0)
        return [f"x{i + 1}" for i in range(feature_count)]

    @property
    def defaults(self) -> list[float]:
        means = getattr(self.scaler, "mean_", None)
        if means is not None:
            return [float(value) for value in means]
        return [0.0 for _ in self.feature_names]

    def payload_lookup(self, payload: dict[str, Any]) -> dict[str, Any]:
        lookup = {}
        for key, value in payload.items():
            lookup[normalize_key(key)] = value
        return lookup

    def find_payload_value(self, feature: str, lookup: dict[str, Any]) -> Any:
        candidates = [feature]
        candidates.extend(MODEL_ALIASES.get(self.key, {}).get(feature, []))
        for candidate in candidates:
            normalized = normalize_key(candidate)
            if normalized in lookup:
                return lookup[normalized]
        return None

    def build_input(self, payload: dict[str, Any]) -> np.ndarray:
        lookup = self.payload_lookup(payload)
        values = []
        feature_names = self.feature_names
        defaults = self.defaults

        for index, feature in enumerate(feature_names):
            default = defaults[index] if index < len(defaults) else 0.0
            raw_value = self.find_payload_value(feature, lookup)
            encoder = self.feature_encoders.get(feature)

            if encoder:
                values.append(encode_value(encoder, raw_value, default))
            else:
                values.append(to_float(raw_value, default))

        return np.array([values], dtype=float)

    def decode_prediction(self, prediction: Any) -> str:
        value = scalar(prediction)
        if self.target_encoder is not None:
            try:
                label = str(self.target_encoder.inverse_transform([value])[0])
                return OUTPUT_LABELS.get(self.key, {}).get(label, label)
            except Exception:
                pass
        if self.key in PREDICTION_LABELS:
            try:
                return PREDICTION_LABELS[self.key][int(value)]
            except (KeyError, TypeError, ValueError):
                pass
        label = str(value)
        return OUTPUT_LABELS.get(self.key, {}).get(label, label)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.load()
        input_data = self.build_input(payload)
        scaled_data = self.scaler.transform(input_data)

        prediction = self.model.predict(scaled_data)[0]
        label = self.decode_prediction(prediction)
        probability = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(scaled_data)[0]
            probability = round(float(np.max(proba)) * 100, 2)

        return {
            "prediction": label,
            "predicted_class": label,
            "probability": probability,
            "confidence": probability,
        }


ML_MODELS = {
    "kidney": MLModel(
        "kidney",
        ML_MODELS_DIR / "kidney" / "kidney_model.pkl",
        ML_MODELS_DIR / "kidney" / "kidney_scaler.pkl",
    ),
    "liver": MLModel(
        "liver",
        ML_MODELS_DIR / "liver" / "xgb_model.pkl",
        ML_MODELS_DIR / "liver" / "scaler.pkl",
        feature_encoders_path=ML_MODELS_DIR / "liver" / "feature_encoders.pkl",
        target_encoder_path=ML_MODELS_DIR / "liver" / "target_encoder.pkl",
    ),
    "lung": MLModel(
        "lung",
        ML_MODELS_DIR / "lung" / "lung_cancer_xgb_model.pkl",
        ML_MODELS_DIR / "lung" / "lung_scaler.pkl",
    ),
    "alzheimer": MLModel(
        "alzheimer",
        ML_MODELS_DIR / "alzheimer" / "alzheimer_model.pkl",
        ML_MODELS_DIR / "alzheimer" / "alzheimer_scaler.pkl",
        feature_encoders_path=ML_MODELS_DIR / "alzheimer" / "alzheimer_label_encoders.pkl",
        target_encoder_path=ML_MODELS_DIR / "alzheimer" / "alzheimer_target_encoder.pkl",
    ),
    "epileptic": MLModel(
        "epileptic",
        ML_MODELS_DIR / "epileptic" / "epileptic_rf_model.pkl",
        ML_MODELS_DIR / "epileptic" / "epileptic_scaler.pkl",
    ),
}


_chatbot_session_class = None
_chatbot_load_error: str | None = None
_chat_sessions: dict[str, dict[str, Any]] = {}
_chat_sessions_lock = threading.Lock()
CHAT_SESSION_TTL_SECONDS = 30 * 60


class FallbackMedicalChatSession:
    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id

    async def send_message(self, message: str) -> dict[str, Any]:
        text = build_fallback_chat_response(message)
        return {"text": text, "report": None, "sources": []}


def is_arabic_text(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def build_general_fallback_response(message: str, lowered: str, arabic: bool) -> str:
    topic_rules = [
        (
            ["headache", "migraine"],
            ["صداع", "راس", "رأس", "دماغ"],
            (
                "Headache is commonly related to stress, dehydration, lack of sleep, eye strain, sinus symptoms, or migraine. Rest, fluids, and avoiding bright light may help. Go to urgent care if it is sudden and severe, follows head trauma, comes with weakness, confusion, fainting, fever with stiff neck, or vision/speech changes. How long has the headache been going on?"
            ),
            (
                "الصداع ممكن يحصل بسبب إجهاد، قلة نوم، جفاف، مشاكل نظر، جيوب أنفية، أو صداع نصفي. جرّب الراحة وشرب سوائل وتقليل الإضاءة لو الصداع مزعج. لو الصداع مفاجئ وشديد جدًا، أو بعد خبطة في الرأس، أو مع ضعف/تنميل/تشوش/إغماء/سخونية مع تيبس رقبة، اتجه للطوارئ. بقاله قد إيه الصداع؟"
            ),
        ),
        (
            ["fever", "temperature"],
            ["سخونية", "حرارة", "حمى"],
            (
                "Fever is usually a sign your body is fighting an infection. Fluids, rest, and monitoring the temperature are reasonable first steps. Seek urgent care for breathing trouble, confusion, stiff neck, severe dehydration, a rash that does not fade, or fever lasting more than a few days. Do you have cough, sore throat, burning urination, or abdominal pain with it?"
            ),
            (
                "السخونية غالبًا علامة إن الجسم بيقاوم عدوى. اشرب سوائل، ارتاح، وتابع درجة الحرارة. لازم رعاية عاجلة لو في ضيق نفس، تشوش، تيبس رقبة، جفاف شديد، طفح مش بيختفي بالضغط، أو الحرارة مستمرة أكتر من كام يوم. هل معها كحة، التهاب حلق، حرقان بول، أو ألم بطن؟"
            ),
        ),
        (
            ["cough", "sore throat", "flu", "cold"],
            ["كحة", "سعال", "زكام", "برد", "حلق", "احتقان"],
            (
                "Cough and throat symptoms are often viral, but can also come from allergies, asthma, reflux, or a chest infection. Warm fluids, rest, and avoiding smoke can help. Seek care quickly if there is shortness of breath, chest pain, blue lips, coughing blood, or high fever. How many days have you had it?"
            ),
            (
                "الكحة أو التهاب الحلق غالبًا بيكونوا بسبب دور برد فيروسي، لكن ممكن كمان حساسية، ربو، ارتجاع، أو التهاب صدري. السوائل الدافئة والراحة وتجنب الدخان بيساعدوا. راجع طبيب بسرعة لو في ضيق نفس، ألم صدر، زرقة، دم مع الكحة، أو سخونية عالية. بقالها كام يوم؟"
            ),
        ),
        (
            ["stomach", "abdominal", "diarrhea", "vomit", "nausea"],
            ["معدة", "بطن", "اسهال", "إسهال", "ترجيع", "قيء", "غثيان"],
            (
                "Stomach symptoms can come from indigestion, gastroenteritis, food poisoning, reflux, or inflammation. Small sips of fluids and light food can help if symptoms are mild. Urgent care is needed for severe or worsening pain, repeated vomiting, blood in stool or vomit, black stool, fainting, or signs of dehydration. Where exactly is the pain?"
            ),
            (
                "أعراض المعدة والبطن ممكن تكون من عسر هضم، نزلة معوية، تسمم غذائي، ارتجاع، أو التهاب. لو الأعراض خفيفة، السوائل على رشفات وأكل خفيف ممكن يساعدوا. الطوارئ مهمة لو الألم شديد أو بيزيد، ترجيع متكرر، دم في البراز أو القيء، براز أسود، إغماء، أو جفاف. الألم فين بالضبط؟"
            ),
        ),
        (
            ["dizzy", "dizziness", "vertigo"],
            ["دوخة", "دوار", "دايخ"],
            (
                "Dizziness may happen with dehydration, low blood sugar, inner-ear problems, anemia, blood pressure changes, or anxiety. Sit down, avoid driving, and drink fluids if you can. Go to emergency care if it comes with chest pain, fainting, severe headache, weakness on one side, slurred speech, or trouble walking. Is it spinning vertigo or lightheadedness?"
            ),
            (
                "الدوخة ممكن تحصل من جفاف، انخفاض سكر، مشكلة في الأذن الداخلية، أنيميا، تغير ضغط، أو قلق. اقعد وتجنب السواقة واشرب سوائل لو تقدر. اتجه للطوارئ لو معها ألم صدر، إغماء، صداع شديد، ضعف في ناحية، لخبطة كلام، أو صعوبة مشي. الدوخة إحساس دوران ولا خفة رأس؟"
            ),
        ),
        (
            ["rash", "skin", "itch"],
            ["طفح", "جلد", "حكة", "هرش", "حساسية"],
            (
                "Skin rash can be allergy, irritation, infection, eczema, or a medication reaction. Avoid scratching and new products until it is clearer. Seek urgent care if there is swelling of lips/face, breathing trouble, fever, rapidly spreading rash, severe pain, or blisters. Did it start after a new food, medicine, or skin product?"
            ),
            (
                "الطفح أو الحكة ممكن يكونوا حساسية، تهيج، عدوى، إكزيما، أو تفاعل مع دواء. بلاش حكّ أو منتجات جديدة لحد ما الصورة توضح. رعاية عاجلة لو في تورم شفايف/وش، ضيق نفس، سخونية، انتشار سريع، ألم شديد، أو فقاقيع. ظهر بعد أكل أو دواء أو منتج جلد جديد؟"
            ),
        ),
        (
            ["back pain", "neck pain", "joint", "knee", "shoulder"],
            ["ظهر", "رقبة", "مفصل", "ركبة", "كتف", "عظام"],
            (
                "Muscle or joint pain is often strain or inflammation, but injury, nerve pressure, or infection can also cause it. Rest the area, avoid heavy effort, and use gentle movement as tolerated. Urgent care is needed after major trauma, or with weakness, numbness, fever, loss of bladder/bowel control, or severe worsening pain. Did it start after an injury or heavy lifting?"
            ),
            (
                "ألم العضلات أو المفاصل غالبًا شد أو التهاب، لكن ممكن إصابة، ضغط على عصب، أو عدوى. ريّح المكان وتجنب المجهود الشديد مع حركة خفيفة حسب قدرتك. الطوارئ مهمة بعد إصابة قوية، أو لو في ضعف/تنميل، سخونية، فقدان تحكم في البول أو البراز، أو ألم بيزيد بشدة. بدأ بعد إصابة أو شيل حاجة تقيلة؟"
            ),
        ),
        (
            ["cancer", "tumor", "brain tumor"],
            ["سرطان", "ورم", "اورام", "أورام"],
            (
                "Cancer or tumor concerns need proper medical evaluation; symptoms alone cannot confirm or rule them out. A doctor may need an exam, imaging, or lab tests depending on the symptom. Seek urgent care for new seizures, sudden weakness, severe persistent headache, unexplained weight loss with worsening symptoms, or repeated vomiting. What symptom or test result made you worry about this?"
            ),
            (
                "القلق من الأورام أو السرطان محتاج تقييم طبي واضح؛ الأعراض لوحدها ما تثبتش ولا تنفي. الطبيب ممكن يحتاج كشف، تحاليل، أو أشعة حسب الشكوى. اتجه لرعاية عاجلة لو في تشنجات جديدة، ضعف مفاجئ، صداع شديد مستمر، نقص وزن غير مفسر مع أعراض بتزيد، أو قيء متكرر. إيه العرض أو نتيجة الفحص اللي خلتك تقلق؟"
            ),
        ),
    ]

    for english_terms, arabic_terms, english_response, arabic_response in topic_rules:
        if any(term in lowered for term in english_terms) or any(
            term in message for term in arabic_terms
        ):
            return arabic_response if arabic else english_response

    if any(char.isalpha() for char in lowered) or arabic:
        return (
            "أقدر أساعدك بمعلومات طبية عامة بناءً على سؤالك. لو بتسأل عن عرض معين، أهم حاجات تحدد الخطورة هي مدة العرض، شدته، هل بيزيد، وهل معه علامات خطر زي ضيق نفس، ألم صدر شديد، إغماء، ضعف مفاجئ، نزيف شديد، أو تشوش؛ في الحالات دي اتجه للطوارئ فورًا. اكتب لي العرض الأساسي أو اسم الحالة اللي تقصدها، واذكر بقاله قد إيه."
            if arabic
            else "I can help with general medical guidance for your question. If this is about a symptom, the key risk details are duration, severity, whether it is worsening, and red flags like shortness of breath, severe chest pain, fainting, sudden weakness, heavy bleeding, or confusion; those need emergency care. Tell me the main symptom or condition and how long it has been going on."
        )

    return (
        "اكتب سؤالك الطبي أو العرض اللي عندك بكلمات واضحة، وأنا هساعدك بمعلومات عامة وخطوات مناسبة."
        if arabic
        else "Type your medical question or symptom clearly, and I will help with general guidance."
    )


def build_fallback_chat_response(message: str) -> str:
    message = (message or "").strip()
    lowered = message.lower()
    arabic = is_arabic_text(message)

    if not message:
        return "اكتب سؤالك الطبي وسأساعدك." if arabic else "Type your medical question and I will help."

    if any(word in lowered for word in ["hello", "hi", "hey"]) or any(
        word in message for word in ["سلام", "اهلا", "أهلا", "هاي"]
    ):
        return (
            "أهلًا، أنا مساعد طبي. اقدر أساعدك بمعلومات عامة عن الأعراض، الوقاية، ومتى تحتاج تراجع طبيب. "
            "اكتب العرض أو السؤال اللي عندك."
            if arabic
            else "Hi, I can help with general medical information, symptoms, prevention, and when to see a doctor. Tell me what is going on."
        )

    if "diabetes" in lowered or "سكر" in message:
        return (
            "أعراض ارتفاع السكر ممكن تشمل عطش شديد، تبول متكرر، إرهاق، زغللة، وجوع زائد. "
            "لو الأعراض مستمرة أو عندك قياسات سكر عالية، الأفضل تعمل تحليل سكر صائم و HbA1c وتراجع طبيب باطنة. "
            "لو فيه قيء شديد، دوخة قوية، أو تشوش وعي فده يحتاج طوارئ."
            if arabic
            else "Common diabetes symptoms include excessive thirst, frequent urination, fatigue, blurred vision, and increased hunger. If symptoms persist or readings are high, check fasting glucose and HbA1c and see a clinician. Severe vomiting, marked dizziness, or confusion needs urgent care."
        )

    if "heart" in lowered or "قلب" in message:
        return (
            "للوقاية من أمراض القلب: قلل التدخين والدهون المتحولة، حافظ على ضغط وسكر وكوليسترول مناسبين، امشِ بانتظام، واهتم بالوزن والنوم. "
            "ألم صدر ضاغط، ضيق نفس شديد، أو ألم ممتد للذراع/الفك يحتاج طوارئ فورًا."
            if arabic
            else "Heart disease prevention includes avoiding smoking, limiting trans fats, controlling blood pressure, blood sugar and cholesterol, regular walking, healthy weight, and good sleep. Crushing chest pain, severe shortness of breath, or pain spreading to the arm or jaw needs emergency care."
        )

    if "covid" in lowered or "كورونا" in message:
        return (
            "أعراض كوفيد قد تشمل حرارة، كحة، التهاب حلق، إرهاق، آلام بالجسم، فقدان شم أو تذوق، وأحيانًا ضيق نفس. "
            "اعزل نفسك لو الأعراض حديثة، واشرب سوائل، واطلب رعاية عاجلة لو عندك ضيق نفس، زرقة، ألم صدر، أو تشوش."
            if arabic
            else "COVID-19 symptoms can include fever, cough, sore throat, fatigue, body aches, loss of smell or taste, and sometimes shortness of breath. Isolate if symptoms are new, stay hydrated, and seek urgent care for shortness of breath, bluish lips, chest pain, or confusion."
        )

    if "blood pressure" in lowered or "ضغط" in message:
        return (
            "ارتفاع الضغط غالبًا لا يسبب أعراض واضحة، لكنه ممكن يسبب صداع أو زغللة أو دوخة عند بعض الناس. "
            "الأهم قياس الضغط أكثر من مرة في أوقات مختلفة. لو القراءة 180/120 أو أعلى، أو معها ألم صدر/ضيق نفس/ضعف مفاجئ، توجه للطوارئ."
            if arabic
            else "High blood pressure often has no clear symptoms, though some people notice headache, blurred vision, or dizziness. The key is repeated measurements at different times. A reading of 180/120 or higher, or symptoms like chest pain, shortness of breath, or sudden weakness, needs emergency care."
        )

    return build_general_fallback_response(message, lowered, arabic)


def get_chatbot_session_class():
    global _chatbot_session_class, _chatbot_load_error
    if _chatbot_session_class is not None:
        return _chatbot_session_class
    if _chatbot_load_error is not None:
        return None

    try:
        from dotenv import load_dotenv

        load_dotenv(CHATBOT_DIR / ".env")
        if str(CHATBOT_DIR) not in sys.path:
            sys.path.insert(0, str(CHATBOT_DIR))

        for module_name in ("chatbot_engine", "key_manager"):
            sys.modules.pop(module_name, None)

        from chatbot_engine import MedicalChatSession

        _chatbot_session_class = MedicalChatSession
        return _chatbot_session_class
    except Exception as exc:
        for module_name in ("chatbot_engine", "key_manager"):
            sys.modules.pop(module_name, None)
        _chatbot_load_error = str(exc)
        return None


def cleanup_chat_sessions() -> None:
    now = time.time()
    with _chat_sessions_lock:
        expired = [
            session_id
            for session_id, entry in _chat_sessions.items()
            if now - entry["last_active"] > CHAT_SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            del _chat_sessions[session_id]


def unavailable_chatbot_text() -> str:
    reason = _chatbot_load_error or "unknown startup error"
    return (
        "الشات بوت متوصل بالموقع، لكنه محتاج مفتاح Gemini قبل ما يرد. "
        "ضيف GEMINI_KEY_1 في ملف chatbot-main/.env ثم أعد تشغيل السيرفر. "
        f"السبب الحالي: {reason}"
    )


@lru_cache(maxsize=2)
def load_keras_model(model_path: str) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelLoadError("Missing Python package while loading CV model: tensorflow") from exc
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load CV model: {exc}") from exc


def preprocess_image(file_storage: Any, image_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ModelLoadError("Missing Python package while processing image: pillow") from exc

    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize(image_size)
    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def binary_image_prediction(
    file_storage: Any,
    *,
    model_path: Path,
    negative_label: str,
    positive_label: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model = load_keras_model(str(model_path))
    image_array = preprocess_image(file_storage)
    raw_prediction = model.predict(image_array)
    score = float(np.ravel(raw_prediction)[0])
    is_positive = score >= threshold
    confidence = score if is_positive else 1 - score
    label = positive_label if is_positive else negative_label
    return {
        "filename": file_storage.filename,
        "prediction": label,
        "predicted_class": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(score, 4),
    }


def latest_file_mtime(path: Path) -> float:
    if path.is_file():
        return path.stat().st_mtime
    mtimes = [item.stat().st_mtime for item in path.rglob("*") if item.is_file()]
    return max(mtimes) if mtimes else 0.0


def prepare_extracted_torch_archive(model_dir: Path) -> Path:
    required_files = ("data.pkl", "version", "byteorder")
    if not all((model_dir / name).exists() for name in required_files):
        raise ModelLoadError(
            "Brain detection model artifact is incomplete. "
            "Expected PyTorch archive files: data.pkl, version, and byteorder."
        )

    cache_dir = Path(tempfile.gettempdir()) / "ai_medical_center_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "brain_detection_last_repacked.pt"
    source_mtime = latest_file_mtime(model_dir)

    if archive_path.exists() and archive_path.stat().st_mtime >= source_mtime:
        return archive_path

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive_root = model_dir.name
        for item in sorted(model_dir.rglob("*")):
            if item.is_file():
                relative_path = item.relative_to(model_dir).as_posix()
                archive.write(item, f"{archive_root}/{relative_path}")

    return archive_path


def find_brain_detection_artifact() -> Path:
    if not BRAIN_DETECTION_DIR.exists():
        raise ModelLoadError(
            "Brain detection model folder was not found: "
            f"{BRAIN_DETECTION_DIR}"
        )

    for pattern in ("*.pt", "*.pth"):
        candidates = sorted(
            BRAIN_DETECTION_DIR.rglob(pattern),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    if BRAIN_DETECTION_EXTRACTED_DIR.exists():
        return prepare_extracted_torch_archive(BRAIN_DETECTION_EXTRACTED_DIR)

    for candidate in sorted(BRAIN_DETECTION_DIR.rglob("*")):
        if candidate.is_dir() and (candidate / "data.pkl").exists():
            return prepare_extracted_torch_archive(candidate)

    raise ModelLoadError(
        "Brain detection model artifact is incomplete. "
        "Add a .pt/.pth YOLO model file, or the extracted PyTorch archive folder."
    )


@lru_cache(maxsize=1)
def load_ultralytics_model(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "Missing Python package while loading brain detection model: "
            "ultralytics. Run pip install -r requirements.txt again."
        ) from exc

    try:
        return YOLO(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load brain detection model: {exc}") from exc


def yolo_brain_detection_prediction(file_storage: Any) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ModelLoadError("Missing Python package while processing image: pillow") from exc

    artifact_path = find_brain_detection_artifact()
    model = load_ultralytics_model(str(artifact_path))
    image = Image.open(file_storage.stream).convert("RGB")
    image_size = {"width": image.width, "height": image.height}
    results = model.predict(source=np.asarray(image), verbose=False)

    names = getattr(model, "names", {}) or {}
    detections = []
    if results:
        result = results[0]
        names = getattr(result, "names", names) or names
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for index in range(len(boxes)):
                class_id = int(boxes.cls[index].item())
                class_name = str(names.get(class_id, class_id))
                confidence = float(boxes.conf[index].item())
                xyxy = [round(float(value), 2) for value in boxes.xyxy[index].tolist()]
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence * 100, 2),
                        "box": xyxy,
                    }
                )

    if not detections:
        return {
            "filename": file_storage.filename,
            "prediction": "No Tumor",
            "predicted_class": "No Tumor",
            "class_name": "No Tumor",
            "confidence": None,
            "tumor_detected": False,
            "detections": [],
            "image_size": image_size,
        }

    best_detection = max(detections, key=lambda item: item["confidence"])
    class_name = best_detection["class_name"]
    is_no_tumor = normalize_label(class_name) == "no tumor"
    prediction = "No Tumor" if is_no_tumor else f"{class_name} Tumor Detected"

    return {
        "filename": file_storage.filename,
        "prediction": prediction,
        "predicted_class": prediction,
        "class_name": class_name,
        "confidence": best_detection["confidence"],
        "tumor_detected": not is_no_tumor,
        "detections": detections,
        "image_size": image_size,
    }


def json_error(message: str, status: int = 400):
    return jsonify({"error": message, "message": message}), status


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/brain_classification.html")
def brain_classification_page():
    return render_template("cv_model.html", **CV_PAGES["brain_classification"])


@app.route("/brain.html")
def brain_detection_page():
    return render_template("cv_model.html", **CV_PAGES["brain"])


@app.route("/chest.html")
def chest_page():
    return render_template("cv_model.html", **CV_PAGES["chest"])


@app.route("/<page_name>.html")
def render_page(page_name: str):
    if page_name not in PAGE_NAMES:
        return json_error("Page not found", 404)
    return render_template(f"{page_name}.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "ml_models": sorted(ML_MODELS),
            "cv_models": ["brain_classification", "brain_detection", "chest"],
            "missing_cv_models": [],
            "chatbot_configured": get_chatbot_session_class() is not None,
            "chatbot_error": _chatbot_load_error,
        }
    )


@app.route("/chat/start", methods=["POST"])
def chat_start():
    cleanup_chat_sessions()
    session_class = get_chatbot_session_class()
    user_id = request.form.get("user_id") or None
    session_id = str(uuid.uuid4())

    session = (
        session_class(user_id=user_id)
        if session_class
        else FallbackMedicalChatSession(user_id=user_id)
    )
    with _chat_sessions_lock:
        _chat_sessions[session_id] = {
            "session": session,
            "last_active": time.time(),
            "lock": threading.Lock(),
        }

    return jsonify(
        {
            "session_id": session_id,
            "configured": session_class is not None,
            "error": _chatbot_load_error,
        }
    )


@app.route("/chat/message", methods=["POST"])
def chat_message():
    session_id = request.form.get("session_id", "")
    message = request.form.get("message", "").strip()
    if not message:
        return json_error("Empty message")

    with _chat_sessions_lock:
        entry = _chat_sessions.get(session_id)
        if entry:
            entry["last_active"] = time.time()

    if not entry:
        session_class = get_chatbot_session_class()
        session = (
            session_class(user_id=None)
            if session_class
            else FallbackMedicalChatSession(user_id=None)
        )
        session_id = session_id or str(uuid.uuid4())
        entry = {
            "session": session,
            "last_active": time.time(),
            "lock": threading.Lock(),
        }
        with _chat_sessions_lock:
            _chat_sessions[session_id] = entry

    session = entry["session"]
    if session is None:
        return jsonify({"text": unavailable_chatbot_text(), "report": None, "sources": []})

    try:
        with entry["lock"]:
            result = asyncio.run(session.send_message(message))
        if not str(result.get("text") or "").strip():
            result["text"] = build_fallback_chat_response(message)
        return jsonify(result)
    except Exception as exc:
        return jsonify(
            {
                "text": f"Chatbot request failed: {exc}",
                "report": None,
                "sources": [],
            }
        )


@app.route("/api/ml/<model_key>/predict", methods=["POST"])
def predict_ml(model_key: str):
    model = ML_MODELS.get(model_key)
    if not model:
        return json_error(f"Unknown ML model: {model_key}", 404)

    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(model.predict(payload))
    except ModelLoadError as exc:
        return json_error(str(exc), 503)
    except Exception as exc:
        return json_error(f"Prediction failed: {exc}", 500)


@app.route("/predict_liver", methods=["POST"])
def predict_liver_legacy():
    return predict_ml("liver")


@app.route("/predict", methods=["POST"])
@app.route("/api/cv/covid/predict", methods=["POST"])
@app.route("/api/cv/pneumonia/predict", methods=["POST"])
@app.route("/api/cv/chest/predict", methods=["POST"])
def predict_chest():
    file_storage = request.files.get("file")
    if not file_storage:
        return json_error("No image file uploaded")
    try:
        return jsonify(
            binary_image_prediction(
                file_storage,
                model_path=MODELS_DIR / "x-ray" / "pneumonia_model.keras",
                negative_label="NORMAL",
                positive_label="PNEUMONIA",
            )
        )
    except ModelLoadError as exc:
        return json_error(str(exc), 503)
    except Exception as exc:
        return json_error(f"Prediction failed: {exc}", 500)


@app.route("/api/cv/brain-classification/predict", methods=["POST"])
def predict_brain_classification():
    file_storage = request.files.get("file")
    if not file_storage:
        return json_error("No image file uploaded")
    try:
        return jsonify(
            binary_image_prediction(
                file_storage,
                model_path=MODELS_DIR
                / "Brain_tumor classification"
                / "brain_tumor_mobilenet.keras",
                negative_label="No Tumor",
                positive_label="Tumor Detected",
            )
        )
    except ModelLoadError as exc:
        return json_error(str(exc), 503)
    except Exception as exc:
        return json_error(f"Prediction failed: {exc}", 500)


@app.route("/api/cv/brain-detection/predict", methods=["POST"])
def predict_brain_detection():
    file_storage = request.files.get("file")
    if not file_storage:
        return json_error("No image file uploaded")
    try:
        return jsonify(yolo_brain_detection_prediction(file_storage))
    except ModelLoadError as exc:
        return json_error(str(exc), 503)
    except Exception as exc:
        return json_error(f"Prediction failed: {exc}", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
