from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class InferenceConfig:
    image_dir: str = "/workspace/HVPG/data/external_validation"
    checkpoint_root: str = "/workspace/HVPG/result/multi_modal_binary/best_model_blip_HVPG_10_swin_hierarchical_ver4"
    results_subdir: str = "external_validation_th2"
    cv_results_subdir: str = "cv_results"
    meta_data_path: str = "/workspace/HVPG/data/10.HVPG-CT_부천순천향data.csv"
    hospital_id_column: str = "병원번호"

    batch_size: int = 3
    fixed_size: Tuple[int, int, int] = (128, 128, 128)
    text_fusion_mode: str = "cross_attn"

    fold_range: Tuple[int, int] = (1, 5)
    fold_cutoff: float = 0.1
    ensemble_cutoff: float = 0.5

    @property
    def results_root_dir(self) -> str:
        return str(Path(self.checkpoint_root) / self.results_subdir)

    @property
    def trained_cv_root(self) -> str:
        return str(Path(self.checkpoint_root) / self.cv_results_subdir)

    @property
    def fold_indices(self) -> Iterable[int]:
        start, end = self.fold_range
        return range(start, end + 1)


def get_inference_config() -> InferenceConfig:
    """Return default inference configuration."""
    return InferenceConfig()

