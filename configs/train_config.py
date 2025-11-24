from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class TrainConfig:
    image_dir: str = "/workspace/HVPG/data/total/"
    checkpoint_root: str = "/workspace/HVPG/result/multi_modal_binary/best_model_blip_HVPG_10_swin_hierarchical_ver_5_year_survival"
    results_subdir: str = "cv_results"
    pretrained_weights_path: str = "/workspace/HVPG/code/multi_modal/pretrained_weights/supervised_suprem_swinunetr_2100.pth"
    meta_data_path: str = "/workspace/HVPG/data/20250921_clinical_meta_data.csv"

    num_classes: int = 1
    batch_size: int = 3
    learning_rate: float = 1e-5
    num_epochs: int = 80
    fixed_size: Tuple[int, int, int] = (128, 128, 128)
    text_fusion_mode: str = "cross_attn"

    test_samples_per_class: int = 60
    random_seed: int = 42
    num_folds: int = 5

    augment_shuffle_prob: float = 0.8
    augment_delete_prob: float = 0.5
    augment_delete_min: float = 0.1
    augment_delete_max: float = 0.3
    augment_min_units: int = 3
    overall_cutoff: float = 0.5

    @property
    def results_root_dir(self) -> str:
        return str(Path(self.checkpoint_root) / self.results_subdir)


def get_train_config() -> TrainConfig:
    """Return default training hyperparameter configuration."""
    return TrainConfig()

