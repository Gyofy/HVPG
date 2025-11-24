from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from utils.text_utils import augment_text, format_patient_text, TEXT_TEMPLATE


class MedicalDataset3D(Dataset):
    def __init__(
        self,
        image_files: Sequence[str],
        label_values: Sequence[Any],
        meta_data,
        image_transforms,
        label_transforms,
        text_template: str = TEXT_TEMPLATE,
        text_augment: bool = False,
        augment_params: Optional[Dict[str, float]] = None,
    ):
        self.image_files = list(image_files)
        self.label_values = list(label_values)
        self.meta_data = meta_data
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms
        self.text_template = text_template
        self.text_augment = text_augment
        self.augment_params = augment_params or {}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        label_value = self.label_values[idx]
        meta_row = self.meta_data.iloc[idx]

        label = torch.tensor(label_value, dtype=torch.float32)
        image = self.image_transforms(img_file)
        label = self.label_transforms(label)

        text_data = format_patient_text(meta_row, self.text_template)
        if self.text_augment:
            text_data = augment_text(text_data, **self.augment_params)

        return image, label, text_data


def custom_collate_fn(batch: List):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels, list(texts)


