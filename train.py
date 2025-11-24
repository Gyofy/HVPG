import os
from glob import glob

from natsort import natsorted
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

from transformers import BlipForImageTextRetrieval, BlipProcessor

from configs import get_train_config  # type: ignore
from utils.transforms import (
    get_inference_image_transforms,
    get_label_transforms,
    get_train_image_transforms,
)
from utils.datasets import MedicalDataset3D, custom_collate_fn
from utils.text_utils import TEXT_TEMPLATE
from utils.text_embeddings import process_text_embeddings
from models.fusion_model import SwinUNETRWithEnhancedText
from utils.metrics import (
    compute_metrics,
    find_best_cutoff_youden,
    save_confusion_matrix,
    save_mean_roc_curve,
    save_overall_metrics,
    save_roc_curve,
    save_text_report,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load_pretrained_backbone(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    store_dict = model.state_dict()
    model_dict = torch.load(checkpoint_path, map_location=device)

    for key in model_dict.keys():
        new_key = ".".join(key.split(".")[1:])
        if "out" in new_key:
            continue
        if "rotation" in new_key:
            continue
        if new_key in store_dict:
            store_dict[new_key] = model_dict[key]

    filtered_dict = {
        k.replace("module.", ""): v
        for k, v in model_dict.items()
        if k.replace("module.", "") in store_dict
    }
    store_dict.update(filtered_dict)
    model.load_state_dict(store_dict, strict=False)


def prepare_dataloader(
    images,
    labels,
    meta,
    image_transforms,
    label_transforms,
    text_augment,
    batch_size,
    augment_params,
):
    dataset = MedicalDataset3D(
        images,
        labels,
        meta,
        image_transforms,
        label_transforms,
        text_template=TEXT_TEMPLATE,
        text_augment=text_augment,
        augment_params=augment_params,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=text_augment,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = get_train_config()
    image_dir = cfg.image_dir
    checkpoint_path = cfg.checkpoint_root
    results_root_dir = cfg.results_root_dir
    os.makedirs(results_root_dir, exist_ok=True)

    model_pre = cfg.pretrained_weights_path
    meta_data = pd.read_csv(cfg.meta_data_path)

    num_classes = cfg.num_classes
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    num_epochs = cfg.num_epochs
    fixed_size = list(cfg.fixed_size)
    text_fusion_mode = cfg.text_fusion_mode

    y = meta_data["hvpgclass"]
    image_files = natsorted(glob(os.path.join(image_dir, "*.nii.gz")))

    class0_indices = np.where(y.values == 0)[0]
    class1_indices = np.where(y.values == 1)[0]
    min_test = cfg.test_samples_per_class
    if len(class0_indices) < min_test or len(class1_indices) < min_test:
        raise ValueError(f"테스트 세트를 위해 각 클래스별 최소 {min_test}개가 필요합니다.")

    rng = np.random.RandomState(cfg.random_seed)
    test_idx0 = rng.choice(class0_indices, size=min_test, replace=False)
    test_idx1 = rng.choice(class1_indices, size=min_test, replace=False)
    test_indices = np.sort(np.concatenate([test_idx0, test_idx1]))
    remaining_indices = np.setdiff1d(np.arange(len(meta_data)), test_indices)

    cv_meta = meta_data.iloc[remaining_indices].reset_index(drop=True)
    cv_images = [image_files[i] for i in remaining_indices]
    cv_labels = y.iloc[remaining_indices].reset_index(drop=True)

    print("-" * 50)
    print("Data set checking")
    print("-" * 50)
    print(
        f"Total samples: {len(meta_data)} (class0={len(class0_indices)}, class1={len(class1_indices)})"
    )
    print(f"Test set: {min_test * 2} (class0={min_test}, class1={min_test})")
    print(f"Remaining for CV: {len(remaining_indices)}")
    print("-" * 50)

    train_image_transforms = get_train_image_transforms(fixed_size)
    val_image_transforms = get_inference_image_transforms(fixed_size)
    label_transforms = get_label_transforms()
    augment_params = dict(
        shuffle_prob=cfg.augment_shuffle_prob,
        delete_prob=cfg.augment_delete_prob,
        deletion_rate_min=cfg.augment_delete_min,
        deletion_rate_max=cfg.augment_delete_max,
        min_units=cfg.augment_min_units,
    )

    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model_blip = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    ).to(device)
    model_blip.eval()
    for param in model_blip.parameters():
        param.requires_grad = False

    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.random_seed)
    fold_fprs, fold_tprs, fold_aucs = [], [], []
    all_cv_labels, all_cv_predictions = [], []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(cv_meta, cv_labels), start=1):
        print(f"Starting Fold {fold_idx}")
        fold_dir = os.path.join(results_root_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train_meta = cv_meta.iloc[train_index].reset_index(drop=True)
        val_meta = cv_meta.iloc[val_index].reset_index(drop=True)
        train_images = [cv_images[i] for i in train_index]
        val_images = [cv_images[i] for i in val_index]
        train_labels = cv_labels.iloc[train_index].reset_index(drop=True)
        val_labels = cv_labels.iloc[val_index].reset_index(drop=True)

        train_loader = prepare_dataloader(
            train_images,
            train_labels,
            train_meta,
            train_image_transforms,
            label_transforms,
            text_augment=True,
            batch_size=batch_size,
            augment_params=augment_params,
        )
        val_loader = DataLoader(
            MedicalDataset3D(
                val_images,
                val_labels,
                val_meta,
                val_image_transforms,
                label_transforms,
                text_template=TEXT_TEMPLATE,
                text_augment=False,
                augment_params=augment_params,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

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
            num_classes=num_classes,
        ).to(device)
        load_pretrained_backbone(model, model_pre, device)

        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        best_val_loss = float("inf")

        train_loss_file = os.path.join(fold_dir, "train_loss.txt")
        val_loss_file = os.path.join(fold_dir, "val_loss.txt")
        metrics_file = os.path.join(fold_dir, "metrics.txt")

        with open(train_loss_file, "w") as f:
            f.write("Epoch\tTrain_Loss\n")
        with open(val_loss_file, "w") as f:
            f.write("Epoch\tVal_Loss\n")

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
            for images, labels, texts in train_pbar:
                images = images.to(device)
                labels = labels.to(device)
                text_embed = process_text_embeddings(texts, processor, model_blip, device, "BLIP")

                optimizer.zero_grad()
                logits, _ = model(images, text_embed)
                batch_loss = loss_function(logits, labels.unsqueeze(1))
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()
                train_pbar.set_postfix({"Batch Loss": f"{batch_loss.item():.4f}"})

            epoch_loss /= len(train_loader)
            print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")
            with open(train_loss_file, "a") as f:
                f.write(f"{epoch + 1}\t{epoch_loss:.4f}\n")

            model.eval()
            val_loss_sum = 0.0
            val_labels_epoch, val_preds_epoch = [], []
            val_pbar = tqdm(val_loader, desc=f"Validation {epoch + 1}/{num_epochs}", leave=True)
            with torch.no_grad():
                for images, labels, texts in val_pbar:
                    images = images.to(device)
                    labels = labels.to(device)
                    text_embed = process_text_embeddings(texts, processor, model_blip, device, "BLIP")
                    logits, _ = model(images, text_embed)
                    batch_loss = loss_function(logits, labels.unsqueeze(1))
                    val_loss_sum += batch_loss.item()
                    val_pbar.set_postfix({"Batch Val Loss": f"{batch_loss.item():.4f}"})

                    preds = torch.sigmoid(logits).cpu().numpy().flatten()
                    val_preds_epoch.extend(preds)
                    val_labels_epoch.extend(labels.cpu().numpy().flatten())

            avg_val_loss = val_loss_sum / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            with open(val_loss_file, "a") as f:
                f.write(f"{epoch + 1}\t{avg_val_loss:.4f}\n")

            try:
                val_auc = roc_auc_score(val_labels_epoch, val_preds_epoch)
            except ValueError:
                val_auc = 0
            print(f"Validation AUC: {val_auc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(fold_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"==> Best model saved to '{best_path}' (Val Loss: {avg_val_loss:.4f})")

            scheduler.step(avg_val_loss)

        best_model_path = os.path.join(fold_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        fold_labels, fold_predictions = [], []
        with torch.no_grad():
            for images, labels, texts in tqdm(val_loader, desc="Inference", leave=True):
                images = images.to(device)
                text_embed = process_text_embeddings(texts, processor, model_blip, device, "BLIP")
                logits, _ = model(images, text_embed)
                predictions = torch.sigmoid(logits).cpu().numpy()
                fold_predictions.extend(predictions.flatten())
                fold_labels.extend(labels.cpu().numpy().flatten())

        fold_labels = np.array(fold_labels)
        fold_predictions = np.array(fold_predictions)
        best_cutoff = find_best_cutoff_youden(fold_labels, fold_predictions)
        cm, report, fpr, tpr, roc_auc = compute_metrics(fold_labels, fold_predictions, best_cutoff)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(roc_auc)

        cm_path = os.path.join(fold_dir, "confusion_matrix.png")
        roc_path = os.path.join(fold_dir, "roc_curve.png")
        save_confusion_matrix(cm, cm_path, title="Confusion Matrix")
        save_roc_curve(fpr, tpr, roc_auc, roc_path, title="Receiver Operating Characteristic")
        save_text_report(report, metrics_file)
        print(f"Fold {fold_idx} evaluation saved in {fold_dir}")

        all_cv_labels.extend(fold_labels.tolist())
        all_cv_predictions.extend(fold_predictions.tolist())

    mean_roc_path = os.path.join(results_root_dir, "cv_mean_roc_curve.png")
    save_mean_roc_curve(fold_fprs, fold_tprs, mean_roc_path)

    overall_cm_path = os.path.join(results_root_dir, "overall_confusion_matrix.png")
    overall_roc_path = os.path.join(results_root_dir, "overall_roc_curve.png")
    overall_metrics_file = os.path.join(results_root_dir, "overall_metrics.txt")
    save_overall_metrics(
        all_cv_labels,
        all_cv_predictions,
        cutoff=cfg.overall_cutoff,
        cm_path=overall_cm_path,
        roc_path=overall_roc_path,
        report_path=overall_metrics_file,
        title_prefix="Overall",
    )

    print("All fold results and overall metrics have been saved.")


if __name__ == "__main__":
    main()


