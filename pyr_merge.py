import cv2
import numpy as np


def make_gaussian_pyramide(im, nlevels=-1):
    pyr = [im.copy()]
    nlevels -= 1

    kernel = np.ones((5, 5), 'uint8')

    while nlevels != 0:
        tmp = cv2.pyrDown(pyr[-1])
        pyr += [tmp.copy()]
        if min(*tmp.shape) <= 2:
            break
        nlevels -= 1
    
    return pyr


def make_laplacian_pyramide(im, nlevels=-1):
    im1 = im.astype(np.int16)
    pyr = []
    nlevels -= 1

    while min(*im1.shape) > 2 and nlevels != 0:
        im2 = cv2.pyrDown(im1)
        im3 = cv2.pyrUp(im2, dstsize=im1.shape[::-1])
        layer = im1 - im3
        pyr += [layer]
        im1 = im2
        nlevels -= 1

    pyr += [im1]
    return pyr


def reconstruct_laplacian_pyramide(pyr, dtype=np.uint8):
    im = pyr[-1]

    for layer in pyr[-2::-1]:
        im = cv2.pyrUp(im, dstsize=layer.shape[::-1])
        im += layer

    t = np.iinfo(dtype)
    im = np.clip(im, t.min, t.max).astype(dtype)
    return im


def add_weighted(u1, u2, w):
    dtype = u1.dtype
    u1 = u1.astype(np.int32)
    u2 = u2.astype(np.int32)
    u = (u1 * (255 - w) + u2 * w) // 255
    return u.astype(dtype)


def merge_pyramides(p1, p2, mask):
    levels = [add_weighted(u1, u2, m) for u1,u2,m in zip(p1, p2, mask)]
    return levels


def dump_pyr_lap(path, pyr):
    for i,l in enumerate(pyr):
        cv2.imwrite(path + f'layer-{i}.png', np.clip(2 * l + 128, 0, 255).astype(np.uint8))


def dump_pyr_gauss(path, pyr):
    for i,l in enumerate(pyr):
        cv2.imwrite(path + f'layer-{i}.png', np.clip(l, 0, 255).astype(np.uint8))


def merge(u1, u2, mask, nlayers=-1):
    assert u1.dtype == np.uint8
    assert u2.dtype == np.uint8
    assert mask.dtype == np.uint8
    assert u1.shape == u2.shape
    assert u1.shape[0:2] == mask.shape

    if nlayers < 0:
        nlayers = int(np.log2(np.min(mask.shape)))

    mask_gpyr = make_gaussian_pyramide(mask)
    dst = np.zeros(u1.shape, dtype=np.uint8)
    for c in range(3):
        u1_lpyr = make_laplacian_pyramide(u1[:,:,c], nlayers)
        u2_lpyr = make_laplacian_pyramide(u2[:,:,c], nlayers)
        dst_lpyr = merge_pyramides(u1_lpyr, u2_lpyr, mask_gpyr)
        channel = reconstruct_laplacian_pyramide(dst_lpyr, dtype=np.int16)
        channel = np.clip(channel, 0, 255).astype(np.uint8)
        dst[:,:,c] = channel

    return dst
