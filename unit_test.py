# Evyatar

import importlib
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

sol2 = importlib.import_module('ex2-danielbrown1.sol2')

REP_GRAY = 1
REP_RGB = 2


def read_image(file_name, representation):
    """
    as in ex1.pdf requirements
    """
    im = imread(file_name)
    im_float = im.astype(np.float64) / 255
    if representation == REP_GRAY:
        im_float = rgb2gray(im_float)
    return im_float


SIGNALS = [
    np.array([[1]], dtype = np.float64),
    np.array([[1], [2]], dtype = np.float64),
    np.array([[1], [2], [1.03]], dtype = np.float64),
    np.array([[1], [2], [100], [4], [200]], dtype = np.float64),
    np.array([[101], [2], [2], [0], [5], [2], [255], [5], [100]], dtype = np.float64),
    np.array([[101], [2], [42], [456], [5], [23], [255], [87], [100]], dtype = np.float64),
    np.array([[101], [2], [2], [0], [255], [0], [255], [0], [255]], dtype = np.float64)
]


def test_dft():
    for i, signal in enumerate(SIGNALS):
        # print(signal)
        bi = np.fft.fft2(signal)
        my = sol2.DFT(signal)
        assert np.allclose(bi, my), "problem with signal number " + str(i) + "\nexpected:\n" + str(
            bi) + '\ngot:\n' + str(my)


FOU_SIGNALS = [
    np.array([[1]], dtype = np.complex128),
    np.array([[1 + 1j], [2]], dtype = np.complex128),
    np.array([[1], [2 + 2j], [3 - 1J]], dtype = np.complex128),
    np.array([[1], [2], [100], [4], [200]], dtype = np.complex128),
    np.array([[101], [2], [2], [0 + 1J], [5], [6J], [255], [5], [100]], dtype = np.complex128),
    np.array([[101], [2], [1.008], [456], [5], [23], [255 - 255J], [87], [100]], dtype = np.complex128),
    np.array([[101], [2], [2], [4 + 2J], [255], [0], [255], [0], [255]], dtype = np.complex128),
    np.array([[374. + 0.j, - 54.7257779 + 284.02597574j,
               - 111.00342312 - 119.6388441j, 200. + 79.67433715j,
               - 216.77079897 + 253.64846163j, - 216.77079897 - 253.64846163j,
               200. - 79.67433715j, - 111.00342312 + 119.6388441j,
               - 54.7257779 - 284.02597574j]], dtype = np.complex128).transpose()
]


def test_idft():
    for i, signal in enumerate(FOU_SIGNALS):
        bi = np.fft.ifft2(signal)
        my = sol2.IDFT(signal)
        assert np.allclose(bi, my), "problem with signal number " + str(i) + "\nexpected:\n" + str(
            bi) + '\ngot:\n' + str(my)


IMAGES = [
    np.array([[1, 1], [2, 2], [1, 1]], dtype = np.float64),
    np.array([[101], [2], [2], [0], [5], [2], [255], [5], [100]], dtype = np.float64),
    np.array([[101, 56], [2, 98], [2, 78], [0, 56], [5, 34], [2, 67], [255, 67], [5, 86], [100, 12]],
             dtype = np.float64),
    np.array([[1, 1, 4, 45], [2, 2, 3, 21], [1, 43, 1, 2], [1, 43, 1, 2]], dtype = np.float64),
    np.array([[1, 1, 3]], dtype = np.float64),
    np.array([[1, 1, 3, 65, 765]], dtype = np.float64),
    np.array([[1, 1, 3, 4, 2, 5, 3, 54], [1, 1, 3, 4, 2, 5, 3, 54]], dtype = np.float64)
]


def test_dft2():
    for image in IMAGES:
        bi = np.fft.fft2(image)
        my = sol2.DFT2(image)
        assert np.allclose(bi, my), '\n' + str(my) + '\n' + str(bi)


F_IMAGES = [
    np.array([[1, 1], [2, 2], [1, 1]], dtype = np.complex128),
    np.array([[101], [2], [2], [0], [5 - 7j], [2], [255], [5], [100]], dtype = np.complex128),
    np.array([[101, 56], [2, 98], [2, 78], [0, 56], [5, 34], [2, 67], [255, 67], [5, 86], [100, 12]],
             dtype = np.complex128),
    np.array([[1, 1, 4, 45], [2, 2 + 2j, 3, 21], [1, 43, 1, 2 + 50j], [1, 43, 1, 2]], dtype = np.complex128),
    np.array([[1, 1, 3]], dtype = np.complex128),
    np.array([[1, 1, 3, 65, 765]], dtype = np.complex128),
    np.array([[1, 1, 3, 4, 2, 5, 3, 54], [1, 1 + 1j, 3, 4, 2, 5, 3, 54]], dtype = np.complex128)
]


def test_idft2():
    for image in F_IMAGES:
        bi = np.fft.ifft2(image)
        my = sol2.IDFT2(image)
        assert np.allclose(bi, my), '\n' + str(my) + '\n' + str(bi)


def show_im(image):
    plt.imshow(image, cmap = 'gray')
    plt.show()


def show_conv_der(image):
    mag = sol2.conv_der(image)
    plt.imshow(mag, cmap = 'gray')
    plt.show()


def show_fourier_der(image):
    four = sol2.fourier_der(image)
    plt.imshow(four, cmap = 'gray')
    plt.show()


def show_blur_spatial(image, size):
    four = sol2.blur_spatial(image, size)
    plt.imshow(four, cmap = 'gray')
    plt.show()


def show_blur_fourier(image, size):
    four = sol2.blur_fourier(image, size)
    plt.imshow(abs(four), cmap = 'gray')
    plt.show()


im1 = read_image('gray.jpg', REP_GRAY)  # path to an image <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

test_dft()
test_idft()
test_dft2()
test_idft2()

# show_im(im1)
# # show_conv_der(im1)
# # show_fourier_der(im1)
# show_blur_spatial(im1, 17)
# show_blur_fourier(im1, 17)
