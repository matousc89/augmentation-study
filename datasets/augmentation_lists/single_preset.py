import cv2 as cv
import albumentations as A


def augmentation_pipeline(fn=None, *, param_count=None):
    def decorate(f):
        def wrapper(*args, **kwargs):
            augmentations = f(*args, **kwargs)  # now just returns a list
            return A.Compose(
                augmentations,
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    clip=True,
                    min_width=1,
                    min_height=1,
                    filter_invalid_bboxes=True,
                ),
            )
        wrapper.__name__ = f.__name__
        return wrapper
    return decorate(fn) if fn else decorate


@augmentation_pipeline(param_count=1)
def get_aug__grid_shuffle(probability): 
    return [
            A.RandomGridShuffle(grid=(1, 2), p=probability),
        ]

@augmentation_pipeline(param_count=1)
def get_aug__horizontal_flip(probability): 
    return [
            A.HorizontalFlip(p=probability),
        ]

@augmentation_pipeline(param_count=1)
def get_aug__vertical_flip(probability): 
    return [
            A.VerticalFlip(p=probability),
        ]

@augmentation_pipeline(param_count=3)
def get_aug__affine_scale(probability): 
    return [
            A.Affine(
                translate_percent=(0, 0),
                scale=(0.8, 1.2),
                rotate=(0, 0),
                shear=(0, 0),
                border_mode=cv.BORDER_CONSTANT,
                p=probability,
            ),
        ]

@augmentation_pipeline(param_count=2)
def get_aug__affine_translate(probability): 
    return [
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(1, 1),
                rotate=(0, 0),
                shear=(0, 0),
                border_mode=cv.BORDER_CONSTANT,
                p=probability,
            ),
        ]

@augmentation_pipeline(param_count=2)
def get_aug__affine_rotate(probability):
    return [
            A.Affine(
                translate_percent=(0, 0),
                scale=1,
                rotate=(-15, 15),
                shear=(0, 0),
                border_mode=cv.BORDER_CONSTANT,
                p=probability,
            ),
        ]

@augmentation_pipeline(param_count=2)
def get_aug__affine_shear(probability):
    return [
            A.Affine(
                translate_percent=(0, 0),
                scale=1,
                rotate=(0, 0),
                shear=(-15, 15),
                border_mode=cv.BORDER_CONSTANT,
                p=probability,
            ),
        ]

@augmentation_pipeline(param_count=3)
def get_aug__elastic_transform(probability):
    return [
            A.ElasticTransform(alpha=500, sigma=250, p=probability),
        ]

@augmentation_pipeline(param_count=3)
def get_aug__grid_distortion(probability):
    return [
            A.GridDistortion(
                num_steps=8,
                distort_limit=0.3,
                p=probability,
            ),
        ]

@augmentation_pipeline(param_count=4)
def get_aug__huesatval(probability):
    return [
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=probability
            ),
        ]

@augmentation_pipeline(param_count=2)
def get_aug__gaussianblur(probability):
    return [
            A.GaussianBlur(sigma_limit=(0, 5), p=probability),
        ]

@augmentation_pipeline(param_count=2)
def get_aug__gaussnoise(probability):
    return [
            A.GaussNoise(std_range=(0.0, 0.25), p=probability),
        ]

@augmentation_pipeline(param_count=3)
def get_aug__coarse_dropout(probability):
    return [
            A.CoarseDropout(
                num_holes_range=(1, 10),  
                hole_height_range=(0.05, 0.3),
                hole_width_range=(0.05, 0.3),
                p=1.0
            ),
        ]

@augmentation_pipeline(param_count=4)
def get_aug__grid_dropout(probability):
    return [
            A.GridDropout(
                ratio=0.5,
                unit_size_range=(10, 100),
                p=probability
            )
        ]
