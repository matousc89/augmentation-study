import albumentations as A


policy_none = A.Compose([
        A.NoOp(p=1.0),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        clip=True,
        min_width=1,
        min_height=1,
        filter_invalid_bboxes=True,
    ),
)


policy_ta_ex = A.Compose([
    A.OneOf([
        # Spatial transforms
        A.Affine(scale=1.0, shear=0, rotate=0, translate_percent=0.2, p=1.0),
        A.Affine(translate_percent=0, shear=0, rotate=0, scale=(0.6, 1.4), p=1.0),
        A.Affine(translate_percent=0, scale=1, rotate=0, shear=(-30, 30), p=1.0),
        A.Affine(translate_percent=0, shear=0, scale=1, rotate=(-30, 30), p=1.0),

        # Dropout transforms
        A.CoarseDropout(
            num_holes_range=(1, 20),  
            hole_height_range=(0.1, 0.6),
            hole_width_range=(0.1, 0.6),
            p=1.0
        ),
        A.GridDropout(
            ratio=0.5,  
            unit_size_range=(10, 100),
            p=1.0
        ),

        # Blur / noise transforms
        A.GaussNoise(std_range=(0.0, 0.5), p=1.0),

        # Color jitter
        A.HueSaturationValue(
            hue_shift_limit=40,
            sat_shift_limit=60, 
            val_shift_limit=40,
            p=1.0
        ),

        # Additional transforms
        A.GridDistortion(num_steps=8, distort_limit=0.6, p=1.0),

        # flip and shuffle
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomGridShuffle(grid=(1, 2), p=1.0),

        # No-op
        A.NoOp(p=1.0),
    ], p=1.0)
],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        clip=True,
        min_width=1,
        min_height=1,
        filter_invalid_bboxes=True,
    )
)


policy_ta_co = A.Compose([
    A.OneOf([
        # Spatial transforms
        A.Affine(scale=1.0, shear=0, rotate=0, translate_percent=0.1, p=1.0),
        A.Affine(translate_percent=0, shear=0, rotate=0, scale=(0.8, 1.2), p=1.0),
        A.Affine(translate_percent=0, scale=1, rotate=0, shear=(-15, 15), p=1.0),
        A.Affine(translate_percent=0, shear=0, scale=1, rotate=(-15, 15), p=1.0),

        # Dropout transforms
        A.CoarseDropout(
            num_holes_range=(1, 10),  
            hole_height_range=(0.05, 0.3),
            hole_width_range=(0.05, 0.3),
            p=1.0
        ),
        A.GridDropout(
            ratio=0.5,  
            unit_size_range=(10, 100),
            p=1.0
        ),

        # Blur / noise transforms
        A.GaussNoise(std_range=(0.0, 0.25), p=1.0),

        # Additional transforms
        A.GridDistortion(num_steps=8, distort_limit=0.3, p=1.0),

        # Color jitter
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30, 
            val_shift_limit=20,
            p=1.0
        ),

        # flip and shuffle
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomGridShuffle(grid=(1, 2), p=1.0),

        # No-op
        A.NoOp(p=1.0),
    ], p=1.0)
],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        clip=True,
        min_width=1,
        min_height=1,
        filter_invalid_bboxes=True,
    )
)
