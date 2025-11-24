from monai.transforms import (
    Compose,
    LoadImage,
    Rand3DElastic,
    RandFlip,
    RandRotate90,
    Resize,
    ScaleIntensity,
    ToTensor,
)


def get_train_image_transforms(fixed_size):
    return Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandFlip(spatial_axis=[0], prob=0.5),
            RandRotate90(prob=1.0, spatial_axes=(1, 0)),
            Rand3DElastic(
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=0.5,
                rotate_range=(0, 0, 0),
                translate_range=(0, 0, 0),
                scale_range=(0, 0, 0),
                mode="bilinear",
                padding_mode="border",
            ),
            Resize(spatial_size=fixed_size),
            ToTensor(),
        ]
    )


def get_inference_image_transforms(fixed_size):
    return Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            Resize(spatial_size=fixed_size),
            ToTensor(),
        ]
    )


def get_label_transforms():
    return Compose([ToTensor()])


