import fourier_merge
import pyr_merge
import numpy as np
import cv2


def copybymask(source, patch, mask):
    dst = source * (1 - mask) + patch * mask
    return dst


def trivial_merge(u1, u2, mask):
    assert u1.dtype == np.uint8
    assert u2.dtype == np.uint8
    assert mask.dtype == np.uint8
    assert u1.shape == u2.shape
    assert u1.shape[0:2] == mask.shape

    mask = np.uint8(mask >= 128)

    u = np.zeros(u1.shape, dtype=np.uint8)
    for c in range(3):
        u[:,:,c] = copybymask(u1[:,:,c], u2[:,:,c], mask)

    return u


def main():
    u1 = cv2.imread('data/sample-1/input-1.jpg')
    u2 = cv2.imread('data/sample-1/input-2.jpg')
    mask = cv2.imread('data/sample-1/mask.jpg', 0)

    u = trivial_merge(u1, u2, mask)
    cv2.imwrite('data/sample-1/out-trivial.png', u)

    u = pyr_merge.merge(u1, u2, mask)
    cv2.imwrite('data/sample-1/out-pyr.png', u)

    u = fourier_merge.merge(u1, u2, mask)
    cv2.imwrite('data/sample-1/out-fourier.png', u)


if __name__ == '__main__':
    main()
