import os
from glob import glob

from natsort import natsorted
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import BlipForImageTextRetrieval, BlipProcessor

from configs import get_inference_config  # type: ignore
from utils.transforms import get_inference_image_transforms, get_label_transforms
from utils.datasets import MedicalDataset3D, custom_collate_fn
from utils.text_utils import TEXT_TEMPLATE
from utils.text_embeddings import process_text_embeddings
from models.fusion_model import SwinUNETRWithEnhancedText
from utils.metrics import (
    compute_metrics,
    save_confusion_matrix,
    save_mean_roc_curve,
    save_overall_metrics,
    save_roc_curve,
    save_text_report,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def extract_hospital_id(filename: str) -> str:
    base = os.path.basename(filename)
    hosp_id = base.split(".")[0]
    return str(hosp_id)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = get_inference_config()
    image_dir = cfg.image_dir
    checkpoint_path = cfg.checkpoint_root
    results_root_dir = cfg.results_root_dir
    os.makedirs(results_root_dir, exist_ok=True)

    trained_cv_root = cfg.trained_cv_root
    meta_data = pd.read_csv(cfg.meta_data_path)

    batch_size = cfg.batch_size
    fixed_size = list(cfg.fixed_size)
    text_fusion_mode = cfg.text_fusion_mode

    image_files = natsorted(glob(os.path.join(image_dir, "*.nii.gz")))
    image_hosp_ids = [extract_hospital_id(f) for f in image_files]

    hospital_col = cfg.hospital_id_column
    if hospital_col not in meta_data.columns:
        raise ValueError(f"meta_data에 병원번호 컬럼({hospital_col})이 없습니다.")
    meta_hosp_ids = meta_data[hospital_col].astype(str)

    common_hosp_ids = set(image_hosp_ids) & set(meta_hosp_ids)
    cv_images, cv_indices = [], []
    for idx, (img, hosp_id) in enumerate(zip(image_files, image_hosp_ids)):
        if hosp_id in common_hosp_ids:
            cv_images.append(img)
            cv_indices.append(idx)

    cv_meta = meta_data[meta_hosp_ids.isin(common_hosp_ids)].reset_index(drop=True)
    cv_labels = cv_meta["hvpgclass"]

    print("-" * 50)
    print("Data set checking")
    print("-" * 50)
    print(f"Total samples: {len(meta_data)}")
    print(f"CV samples: {len(cv_meta)}")
    print("-" * 50)

    image_transforms = get_inference_image_transforms(fixed_size)
    label_transforms = get_label_transforms()
    test_dataset = MedicalDataset3D(
        cv_images,
        cv_labels,
        cv_meta,
        image_transforms,
        label_transforms,
        text_template=TEXT_TEMPLATE,
        text_augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model_blip = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    ).to(device)
    model_blip.eval()
    for param in model_blip.parameters():
        param.requires_grad = False

    all_cv_labels = None
    fold_predictions_list = []
    fold_fprs, fold_tprs, fold_aucs = [], [], []

    for fold_idx in cfg.fold_indices:
        print(f"Starting Fold {fold_idx}")
        fold_dir = os.path.join(results_root_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        model = SwinUNETRWithEnhancedText(
            img_size=fixed_size,
            in_channels=1,
            out_channels=1,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            spatial_dims=3,
            fusion_mode=text_fusion_mode,
            num_classes=1,
        ).to(device)

        trained_fold_ckpt = os.path.join(trained_cv_root, f"fold_{fold_idx}", "best_model.pth")
        if not os.path.exists(trained_fold_ckpt):
            raise FileNotFoundError(f"Fold checkpoint missing: {trained_fold_ckpt}")

        state_dict = torch.load(trained_fold_ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        fold_labels, fold_predictions = [], []
        with torch.no_grad():
            for images, labels, texts in tqdm(test_loader, desc="Inference", leave=True):
                images = images.to(device)
                text_embed = process_text_embeddings(texts, processor, model_blip, device, "BLIP")
                logits, _ = model(images, text_embed)
                predictions = torch.sigmoid(logits).cpu().numpy()
                fold_predictions.extend(predictions.flatten())
                fold_labels.extend(labels.cpu().numpy().flatten())

        fold_predictions = np.array(fold_predictions)
        fold_labels = np.array(fold_labels)
        if all_cv_labels is None:
            all_cv_labels = fold_labels.copy()

        best_cutoff = cfg.fold_cutoff
        cm, report, fpr, tpr, roc_auc = compute_metrics(fold_labels, fold_predictions, best_cutoff)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(roc_auc)

        cm_path = os.path.join(fold_dir, "confusion_matrix.png")
        roc_path = os.path.join(fold_dir, "roc_curve.png")
        save_confusion_matrix(cm, cm_path, title="Confusion Matrix")
        save_roc_curve(fpr, tpr, roc_auc, roc_path, title="Receiver Operating Characteristic")
        fold_metrics_file = os.path.join(fold_dir, "metrics.txt")
        save_text_report(report, fold_metrics_file)

        print(f"Fold {fold_idx} evaluation saved in {fold_dir}")
        fold_predictions_list.append(fold_predictions)

    mean_roc_path = os.path.join(results_root_dir, "cv_mean_roc_curve.png")
    save_mean_roc_curve(fold_fprs, fold_tprs, mean_roc_path)

    avg_preds = np.mean(np.vstack(fold_predictions_list), axis=0)
    best_cutoff = cfg.ensemble_cutoff
    overall_cm_path = os.path.join(results_root_dir, "overall_confusion_matrix.png")
    overall_roc_path = os.path.join(results_root_dir, "overall_roc_curve.png")
    overall_metrics_file = os.path.join(results_root_dir, "overall_metrics.txt")
    save_overall_metrics(
        all_cv_labels,
        avg_preds,
        cutoff=best_cutoff,
        cm_path=overall_cm_path,
        roc_path=overall_roc_path,
        report_path=overall_metrics_file,
        title_prefix="Overall",
    )

    print("All fold results and overall metrics have been saved.")


if __name__ == "__main__":
    main()


