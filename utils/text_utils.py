import random
import re
from typing import Dict

# -----------------------------------------------------------------------------
# Text template & utilities
# -----------------------------------------------------------------------------

TEXT_TEMPLATE = (
    "This patient is a {Sex} aged {Age} years. The blood test results show "
    "Albumin {Albumin} g/dL, Total Bilirubin {TB} mg/dL, AST {AST} U/L, "
    "ALT {ALT} U/L, WBC {WBC}/μL, PMN {PMN} %, Hemoglobin {Hg} g/dL, "
    "Platelet count {PLT}/μL, Sodium {Na} mEq/L, and Creatinine {Cr} mg/dL. "
    "Additionally, the patient has a liver/spleen volume ratio of "
    "{volume_ratio:.3f}, a liver volume of {liver_volume:.3f} cm³, and a "
    "spleen volume of {spleen_volume:.3f} cm³."
)


def _split_text_units(text: str):
    units = re.split(r"[\.;]|,", text)
    units = [u.strip() for u in units if u and u.strip()]
    return units


def _join_text_units(units):
    if not units:
        return ""
    text = ", ".join(units)
    if not text.endswith("."):
        text += "."
    return text


def augment_text(
    text: str,
    shuffle_prob: float = 0.5,
    delete_prob: float = 0.5,
    deletion_rate_min: float = 0.1,
    deletion_rate_max: float = 0.3,
    min_units: int = 3,
) -> str:
    units = _split_text_units(text)
    if len(units) <= 1:
        return text

    if random.random() < shuffle_prob:
        random.shuffle(units)

    if random.random() < delete_prob and len(units) > min_units:
        rate = random.uniform(deletion_rate_min, deletion_rate_max)
        num_to_delete = max(1, int(len(units) * rate))
        num_to_delete = min(num_to_delete, max(0, len(units) - min_units))
        if num_to_delete > 0:
            indices = list(range(len(units)))
            delete_idx = set(random.sample(indices, k=num_to_delete))
            units = [u for i, u in enumerate(units) if i not in delete_idx]
    return _join_text_units(units)


def _build_text_payload(meta_row) -> Dict[str, float]:
    return {
        "Albumin": meta_row["Albumin"],
        "TB": meta_row["TB"],
        "AST": meta_row["AST"],
        "ALT": meta_row["ALT"],
        "WBC": meta_row["WBC"],
        "PMN": meta_row["PMN"],
        "Hg": meta_row["Hg"],
        "PLT": meta_row["PLT"],
        "Na": meta_row["Na"],
        "Cr": meta_row["Cr"],
        "volume_ratio": meta_row["volume_ratio"],
        "liver_volume": meta_row["liver_volume"],
        "spleen_volume": meta_row["spleen_volume"],
        "Sex": meta_row["Sex"],
        "Age": meta_row["Age"],
    }


def format_patient_text(meta_row, template: str = TEXT_TEMPLATE) -> str:
    payload = _build_text_payload(meta_row)
    return template.format(**payload)


