import cv2


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=False):
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict):
        im = data['img']
        if self.resize_text_polys:
            text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_text_polys:
                # text_polys *= scale
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data['img'] = im
        if self.resize_text_polys:
            data['text_polys'] = text_polys
        return data