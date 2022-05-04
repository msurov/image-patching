import numpy as np
import cv2
from main import trivial_merge
import matplotlib.pyplot as plt


def brightness_deriv():

    u1 = cv2.imread('data/sample-1/input-1-modified.jpg')
    u2 = cv2.imread('data/sample-1/input-2.jpg')
    mask = cv2.imread('data/sample-1/mask.jpg', 0)

    u = trivial_merge(u1, u2, mask)
    cv2.imwrite('data/sample-1/out-trivial.png', u)

    font = {'size': 18, 'family': 'Times New Roman'}
    
    x = np.arange(600, 800)
    f = u[1275,600:800,0].astype(int)
    fig,axes = plt.subplots(2, 1, sharex=True)
    plt.sca(axes[0])
    plt.plot(x, f)
    plt.ylabel(R'brigtness $\quad u(x)$', fontdict=font)
    plt.grid(True)
    plt.sca(axes[1])
    plt.ylabel(R'derivative $\,\, \frac{d u}{d x}$', fontdict=font)
    plt.plot(x[1:], np.diff(f))
    plt.xlabel('x', fontdict=font)
    plt.grid(True)
    plt.subplots_adjust(0.14, 0.14, 0.98, 0.98, 0.01, 0.03)
    plt.savefig('data/sample-1/brightness_change.png')


def vector_fields():
    u1 = cv2.imread('data/sample-1/input-1.jpg')
    u2 = cv2.imread('data/sample-1/input-2.jpg')
    mask = cv2.imread('data/sample-1/mask.jpg', 0)
    mask = np.uint8(mask >= 128)

    u1 = u1[:,:,0]
    g1 = np.zeros(u1.shape + (3,), dtype=np.int16)
    g1[:,:,0] = cv2.Sobel(u1, cv2.CV_16S, 1, 0)
    g1[:,:,1] = cv2.Sobel(u1, cv2.CV_16S, 0, 1)
    g1 = np.clip(g1 + 128, 0, 255).astype(np.uint8)
    g1 = g1 * (1 - mask[:,:,np.newaxis])

    u2 = u2[:,:,0]
    g2 = np.zeros(u2.shape + (3,), dtype=np.int16)
    g2[:,:,0] = cv2.Sobel(u2, cv2.CV_16S, 1, 0)
    g2[:,:,1] = cv2.Sobel(u2, cv2.CV_16S, 0, 1)
    g2 = np.clip(g2 + 128, 0, 255).astype(np.uint8)
    g2 = g2 * mask[:,:,np.newaxis]

    g = g1 + g2

    cv2.imwrite('data/sample-1/g1.png', g1)
    cv2.imwrite('data/sample-1/g2.png', g2)
    cv2.imwrite('data/sample-1/g.png', g)


def merge_schematic():
    u1 = cv2.imread('data/sample-1/input-1.jpg')
    u2 = cv2.imread('data/sample-1/input-2.jpg')
    mask = cv2.imread('data/sample-1/mask.jpg', 0)
    mask = np.float32(1) * (mask >= 128)
    m1 = (1 - mask)
    u1 = u1 * m1[:,:,np.newaxis]
    cv2.imwrite('data/sample-1/s1.png', u1)

    m2 = mask
    u2 = u2 * m2[:,:,np.newaxis]
    cv2.imwrite('data/sample-1/s2.png', u2)

if __name__ == '__main__':
    merge_schematic()
