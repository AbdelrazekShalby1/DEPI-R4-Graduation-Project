import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_DIR      = os.getenv("PATIENT_MEMORY_DIR", "patient_memory_data")
MAX_VISITS      = 20
MAX_USERS_CACHE = 1000


def _user_path(user_id: str) -> str:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")[:64]
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, f"{safe}.json")


class PatientMemory:
    """
    One JSON file per patient (`user_id`), holding a rolling list of past
    visits. A "visit" == one chat session: it starts empty when the session
    is created and gets updated in place (not appended to) as the
    conversation's clinical_state evolves, so a single visit stays a single
    JSON entry regardless of how many turns the conversation has.
    """

    def __init__(self, user_id: str):
        self.user_id  = user_id
        self._lock    = threading.Lock()
        self._visits: list[dict] = []
        self._dirty   = False
        self._load()

    def _load(self) -> None:
        path = _user_path(self.user_id)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._visits = data.get("visits", [])
            logger.debug("Loaded %d visit(s) for patient %s",
                         len(self._visits), self.user_id)
        except Exception as exc:
            logger.warning("Failed to load patient memory for %s: %s", self.user_id, exc)

    def save(self) -> None:
        if not self._dirty:
            return

        with self._lock:
            snapshot = list(self._visits)
            self._dirty = False

        path = _user_path(self.user_id)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"visits": snapshot}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            logger.debug("Saved patient memory for %s (%d visits)",
                         self.user_id, len(snapshot))
        except Exception as exc:
            with self._lock:
                self._dirty = True
            logger.warning("Failed to save patient memory for %s: %s", self.user_id, exc)

    def start_visit(self) -> str:
        entry = {
            "id":             uuid.uuid4().hex[:8],
            "started_at":     int(time.time()),
            "updated_at":     int(time.time()),
            "clinical_state": {},
            "report":         None,
        }
        with self._lock:
            self._visits.append(entry)
            if len(self._visits) > MAX_VISITS:
                self._visits = self._visits[-MAX_VISITS:]
            self._dirty = True
        self.save()
        return entry["id"]

    def update_visit(self, visit_id: str,
                     clinical_state: Optional[dict] = None,
                     report: Optional[dict] = None) -> bool:
        found = False
        with self._lock:
            for v in self._visits:
                if v["id"] == visit_id:
                    if clinical_state is not None:
                        v["clinical_state"] = clinical_state
                    if report is not None:
                        v["report"] = report
                    v["updated_at"] = int(time.time())
                    self._dirty = True
                    found = True
                    break

        if not found:
            return False

        self.save()
        return True

    def get_recent_visits(self, limit: int = 3,
                          exclude_visit_id: Optional[str] = None) -> list[dict]:
        with self._lock:
            visits = [v for v in self._visits if v["id"] != exclude_visit_id]
        return visits[-limit:]

    def build_context_text(self, lang: str, limit: int = 3,
                           exclude_visit_id: Optional[str] = None) -> str:
        """
        Formats the patient's recent visits into a short block of text meant
        to be injected as context at the start of a new chat session — NOT
        shown to the patient directly, just given to the model so it can
        pick up the conversation with the patient's history in mind.
        """
        visits = self.get_recent_visits(limit, exclude_visit_id)
        visits = [v for v in visits if v.get("clinical_state")]
        if not visits:
            return ""

        lines = []
        for v in visits:
            date_str = time.strftime("%Y-%m-%d", time.localtime(v["updated_at"]))
            cs        = v.get("clinical_state") or {}
            complaint = cs.get("chief_complaint") or ("غير محدد" if lang == "ar" else "unspecified")
            worst     = cs.get("worst_case_considered")
            diffs     = cs.get("differential_so_far") or []
            diff_txt  = ", ".join(d.get("condition", "") for d in diffs if isinstance(d, dict) and d.get("condition"))

            if lang == "ar":
                line = f"- بتاريخ {date_str}: الشكوى \"{complaint}\""
                if diff_txt:
                    line += f"، الاحتمالات اللي اتناقشت: {diff_txt}"
                if worst:
                    line += f"، أخطر احتمال اتفكر فيه: {worst}"
            else:
                line = f"- On {date_str}: complaint was \"{complaint}\""
                if diff_txt:
                    line += f", discussed possibilities: {diff_txt}"
                if worst:
                    line += f", worst case considered: {worst}"
            lines.append(line)

        header = ("[زيارات سابقة لنفس المريض — للسياق فقط، ماتفترضيش إن الحالة اتحلت أو لسه موجودة، اسألي لو محتاجة تتأكدي]"
                  if lang == "ar" else
                  "[Previous visits for this patient — context only, don't assume the issue is resolved or still ongoing, ask if you need to confirm]")
        return header + "\n" + "\n".join(lines)


_users_lock = threading.Lock()
_users: OrderedDict[str, PatientMemory] = OrderedDict()


def get_patient_memory(user_id: str) -> PatientMemory:
    with _users_lock:
        if user_id in _users:
            _users.move_to_end(user_id)
            return _users[user_id]

        try:
            mem = PatientMemory(user_id)
        except Exception as exc:
            logger.error("Failed to create PatientMemory for %s: %s", user_id, exc)
            raise

        _users[user_id] = mem
        _users.move_to_end(user_id)

        while len(_users) > MAX_USERS_CACHE:
            evicted_id, evicted_mem = _users.popitem(last=False)
            if evicted_mem is not None:
                try:
                    evicted_mem.save()
                except Exception as exc:
                    logger.error("Failed to flush evicted patient %s: %s",
                                 evicted_id, exc)
            logger.debug("LRU evicted patient memory: %s", evicted_id)

    return mem


def flush_all() -> None:
    with _users_lock:
        snapshot = list(_users.values())

    saved = 0
    for mem in snapshot:
        if mem is None:
            continue
        try:
            mem.save()
            saved += 1
        except Exception as exc:
            logger.error("flush_all: failed to save patient %s: %s",
                         mem.user_id, exc)

    logger.info("patient_memory: flushed %d patient memories", saved)
