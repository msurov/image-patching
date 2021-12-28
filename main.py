import cv2
import numpy as np


def merge(source, patch, mask, lam = 200000.0):
    Sy = (np.roll(source, -1, axis=0) - np.roll(source, 1, axis=0)) / 2
    Sx = (np.roll(source, -1, axis=1) - np.roll(source, 1, axis=1)) / 2
    Py = (np.roll(patch, -1, axis=0) - np.roll(patch, 1, axis=0)) / 2
    Px = (np.roll(patch, -1, axis=1) - np.roll(patch, 1, axis=1)) / 2

    gx = Px * mask + Sx * (1 - mask)
    gy = Py * mask + Sy * (1 - mask)
    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)
    F = np.fft.fft2(source)

    M,N = np.shape(F)
    n = np.arange(0, N)
    m = np.arange(0, M)
    Dx = 1j * np.sin(2 * np.pi * n[np.newaxis,:] / N)
    Dy = 1j * np.sin(2 * np.pi * m[:,np.newaxis] / M)

    tmp1 = -F + lam * Dx * Gx + lam * Dy * Gy
    tmp2 = lam * (Dx**2 + Dy**2) - 1

    U = tmp1 / tmp2
    u = np.fft.ifft2(U)
    u = u.real
    return u

def copybymask(source, patch, mask):
    dst = source * (1 - mask) + patch * mask
    return dst

def main():
    source = cv2.imread('data/source.png')
    source = source / 255.
    patch = cv2.imread('data/patch.png')
    patch = patch / 255.
    mask = cv2.imread('data/mask.png', 0)
    mask = np.uint8(mask > 0)

    u = np.zeros(source.shape)
    u[:,:,0] = merge(source[:,:,0], patch[:,:,0], mask)
    u[:,:,1] = merge(source[:,:,1], patch[:,:,1], mask)
    u[:,:,2] = merge(source[:,:,2], patch[:,:,2], mask)

    u = np.clip(255 * u, 0, 255)
    u = np.array(u, dtype=np.uint8)
    cv2.imwrite('data/out-merge.png', u)

    u = np.zeros(source.shape)
    u[:,:,0] = copybymask(source[:,:,0], patch[:,:,0], mask)
    u[:,:,1] = copybymask(source[:,:,1], patch[:,:,1], mask)
    u[:,:,2] = copybymask(source[:,:,2], patch[:,:,2], mask)

    u = np.clip(255 * u, 0, 255)
    u = np.array(u, dtype=np.uint8)
    cv2.imwrite('data/out-copy.png', u)


main()
