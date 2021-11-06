import albumentations as A
from albumentations.pytorch import ToTensorV2


class BasicTransform:
    def __init__(self, mean=None, std=None, normalize=False, to_tensor=False):
        alb_fns = []
        if normalize:
            kwargs = dict()
            if mean is not None:
                kwargs['mean'] = mean
            if std is not None:
                kwargs['std'] = std
            alb_fns.append(A.Normalize(**kwargs))

        if to_tensor:
            alb_fns.append(ToTensorV2())

        self.alb_transform_fn = A.Compose(alb_fns)

    def __call__(self, image, bboxes=[], masks=[]):
        final_result = self.alb_transform_fn(image=image)
        return dict(image=final_result['image'], bboxes=bboxes, masks=masks)