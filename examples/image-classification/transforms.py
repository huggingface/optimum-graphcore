from torchvision import transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

def get_transforms(config_name, training_args, feature_extractor):
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    if config_name == "facebook/convnext-tiny-224":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        resize_img = training_args.input_size > 32
        _train_transforms  = create_transform(
                    input_size= training_args.input_size,
                    is_training=True,
                    color_jitter=0.4,
                    auto_augment='rand-m9-mstd0.5-inc1',
                    interpolation='bicubic',
                    re_prob=0.25,
                    re_mode='pixel',
                    re_count=1,
                    mean=mean,
                    std=std,
                )
        t= []
        if resize_img:
            # warping (no cropping) when evaluated at 384 or larger
            if training_args.input_size >= 384:
                t.append(
                transforms.Resize((training_args.input_size, training_args.input_size),
                                interpolation=transforms.InterpolationMode.BICUBIC),
            )
                print(f"Warping {training_args.input_size} size input images...")
            else:
                crop_pct = 224 / 256
                size = int(training_args.input_size / crop_pct)
                t.append(
                    # to maintain same ratio w.r.t. 224 images
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(training_args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        _val_transforms = transforms.Compose(t)

    else:
        #default transforms
        _train_transforms = Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        _val_transforms = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )

    return (_train_transforms, _val_transforms)





