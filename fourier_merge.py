from re import L
import cv2
import numpy as np


def merge_channel(source, patch, mask, lam = 1e+6):
    Sy = (np.roll(source, -1, axis=0) - np.roll(source, 1, axis=0)) / 2
    Sx = (np.roll(source, -1, axis=1) - np.roll(source, 1, axis=1)) / 2
    Py = (np.roll(patch, -1, axis=0) - np.roll(patch, 1, axis=0)) / 2
    Px = (np.roll(patch, -1, axis=1) - np.roll(patch, 1, axis=1)) / 2

    f = source * (1 - mask) + patch * mask

    gx = Px * mask + Sx * (1 - mask)
    gy = Py * mask + Sy * (1 - mask)
    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)
    F = np.fft.fft2(f)

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


def merge(u1, u2, mask):
    assert u1.dtype == np.uint8
    assert u2.dtype == np.uint8
    assert mask.dtype == np.uint8
    assert u1.shape == u2.shape
    assert u1.shape[0:2] == mask.shape

    u1 = u1 / np.float32(255)
    u2 = u2 / np.float32(255)
    mask = np.uint8(mask >= 128)

    u = np.zeros(u1.shape, dtype=np.float32)
    u[:,:,0] = merge_channel(u1[:,:,0], u2[:,:,0], mask)
    u[:,:,1] = merge_channel(u1[:,:,1], u2[:,:,1], mask)
    u[:,:,2] = merge_channel(u1[:,:,2], u2[:,:,2], mask)
    u = np.clip(255 * u, 0, 255).astype(np.uint8)
    return u
