import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import img_as_float64
from scipy.misc import imread

# ================
# Global variables
# ================
# size of rgb's color dimensions
RGB_SIZE = 3
# size of gray scale's color dimensions
GRAY_SCALE_SIZE = 1
# index of the Y channel
Y_INDEX = 0
# constant to normalize a [0,255] image
NORMALIZE_CONST = 255
# representation code for RGB image
RGB_REPRESENTATION = 2
# matrix for converting RGB image to YIQ
RGB_2_YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
# array of all natural numbers [0,256)
GRAY_RANGE = np.arange(NORMALIZE_CONST + 1)

# ==========
# Functions
# ==========


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: The filename of an image on disk (could be grayscale or RGB).
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
            image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with intensities normalized to [0,1]
    """
    rgb_img = imread(filename)
    rgb_img = img_as_float64(rgb_img)
    if representation == GRAY_SCALE_SIZE:
        rgb_img = rgb2gray(rgb_img)
    return rgb_img / NORMALIZE_CONST


def imdisplay(filename, representation):
    """
    Utilizes read_image function to display an image in a given representation. Displays the image.
    :param filename: The filename of an image on disk (could be grayscale or RGB).
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
            image (1) or an RGB image (2).
    """
    img = read_image(filename, representation)
    plt.figure()
    if representation == GRAY_SCALE_SIZE:
        plt.imshow(img, cmap = "gray")
    else:
        plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space.
    :param imRGB: an RGB image, a heightXwidthX3 np.float64 matrices.
    :return: the YIQ representation of the image.
    """
    return imRGB.dot(RGB_2_YIQ_MATRIX).clip(min = -1, max = 1)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space.
    :param imYIQ: a YIQ image, a heightXwidthX3 np.float64 matrices.
    :return: the RGB representation of the image.
    """
    return imYIQ.dot(np.linalg.inv(RGB_2_YIQ_MATRIX)).clip(min = 0, max = 1)


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: The input grayscale or RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
        im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    if len(im_orig.shape) == RGB_SIZE:
        yiq = rgb2yiq(im_orig)
        y_channel = yiq[:, :, Y_INDEX] * NORMALIZE_CONST
    else:
        y_channel = im_orig * NORMALIZE_CONST
    hist_orig = np.histogram(y_channel, bins = NORMALIZE_CONST + 1, range = [0, NORMALIZE_CONST + 1])[0]
    f, ex = plt.subplots(4, sharex = 'row')
    # f.subplot_adjust(hspace = 0.3)
    first_gray = y_channel.min()
    cumulative_hist = hist_orig.cumsum()
    normal_factor = NORMALIZE_CONST / y_channel.size
    cum_normalized = ((cumulative_hist - first_gray) * normal_factor).astype(int)
    equalized_y = cum_normalized[y_channel.astype(int)]
    hist_eq = np.histogram(equalized_y, bins = NORMALIZE_CONST + 1, range = [0, NORMALIZE_CONST + 1])[0]
    ex[0].plot(hist_orig), ex[0].set_title('Histogram')
    ex[1].plot(cumulative_hist), ex[1].set_title('Cumulative Histogram')
    ex[2].plot(cum_normalized), ex[2].set_title('Cum Histogram Norm')
    ex[3].plot(hist_eq), ex[3].set_title('Hist Eq')
    # ex[4].plot(hist_eq), ex[4].set_title('Hist Eq')
    if len(im_orig.shape) == RGB_SIZE:
        im_eq = yiq
        im_eq[:, :, Y_INDEX] = equalized_y / NORMALIZE_CONST
        im_eq = yiq2rgb(im_eq)
    else:
        im_eq = y_channel / NORMALIZE_CONST
    return [im_eq, hist_orig, hist_eq]


def calc_q(histogram, z, i):
    """
    An aid function for the quantization function. calculates the q array in each iteration.
    :param histogram: the histogram of the original image.
    :param z: the z array in the current iteration.
    :param i: the index of which to calculate q[i]
    :return: the fixed q[i] value
    """
    left_limit = z[i]
    right_limit = z[i + 1]
    range_arr = GRAY_RANGE[left_limit:right_limit]
    local_sum = sum(range_arr * histogram[left_limit:right_limit])
    divider = sum(histogram[left_limit:right_limit])
    if divider == 0:
        return divider
    return local_sum / divider


def calc_error(histogram, q, z):
    """
    calculates the error in each iteration.
    :param histogram: the histogram of the original image
    :param q: the q array in the current iteration.
    :param z: the z array in the current iteration.
    :return: the error in this iteration
    """
    local_error = 0
    for i in range(len(q)):
        z_minus_q = (np.arange(z[i], z[i + 1]) - q[i]) ** 2
        local_error += np.dot(histogram[z[i]: z[i + 1]], z_minus_q.T)
    return local_error


def check_delta_z(old_z, new_z):
    """
    An aid function for the quantization process. checks whether the z array has changed since the last iteration.
    :param old_z: the z array of last iteration
    :param new_z: the new z array
    :return: true iff the z array hasn't changed, false otherwise.
    """
    for i in range(len(new_z)):
        if old_z[i] != new_z[i]:
            return False
    return True


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image.
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of intensities your output im_quant image should have.
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: a list [im_quant, error] where:
        im_quant - is the quantized output image.
        error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
        quantization procedure.
    """
    if len(im_orig.shape)== RGB_SIZE: # if im_orig is RGB
        yiq = rgb2yiq(im_orig)
        y_channel = yiq[:, :, Y_INDEX] * NORMALIZE_CONST
    else: # if im_orig is grayscale
        y_channel = im_orig * NORMALIZE_CONST
    hist_orig, bins = np.histogram(y_channel, bins = 256, range = [0, 256])
    interval_size, p_counter, index = y_channel.size / n_quant, hist_orig[0], 0
    z, old_z, error, q = [0 for _ in range(n_quant + 1)], [0 for _ in range(n_quant + 1)], [], [0 for _ in
                                                                                                range(n_quant)]
    cumulative = hist_orig.cumsum()
    z[0], z[-1] = 0, NORMALIZE_CONST
    for i in range(1, n_quant):
        z[i] = np.where(cumulative > i * interval_size)[0][0]
    # iterating and updating q and z
    for iteration in range(n_iter):
        for i in range(n_quant):
            q[i] = int(calc_q(hist_orig, z, i))
        for i in range(n_quant - 1):
            z[i + 1] = int((q[i] + q[i + 1]) / 2)
        local_error = calc_error(hist_orig, q, z)
        # checking if the error hasn't changed
        if iteration:
            if check_delta_z(z, old_z) | local_error > error[iteration - 1]:
                break
        error.append(local_error)
        old_z = z.copy()
    look_up_table = np.zeros(NORMALIZE_CONST + 1)
    for i in range(len(q)):
        look_up_table[z[i]:z[i + 1]] = q[i]
    quantized_y = look_up_table[y_channel.astype(int)]
    if len(im_orig.shape) == RGB_SIZE: # if im_orig is RGB
        im_quant = yiq
        im_quant[:, :, Y_INDEX] = quantized_y / NORMALIZE_CONST
        im_quant = yiq2rgb(im_quant)
    else:  # if im_orig is grayscale
        im_quant = y_channel / NORMALIZE_CONST
    return im_quant, error


def quantize_rgb(im_orig, n_quant):
    """
    BONUS QUESTION- rgb quantizaion. I implemented it so that I select the q such that it would be the maximal part of
    the middle half of each z interval, because it is important that the different colors would be both maximized
    number-of-pixels wise and both far from each other.
    :param im_orig: the input RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of colors in the quantized image.
    :return: the quantized image.
    """
    im_orig *= NORMALIZE_CONST
    # choosing the dimension- R, G or B- with the widest spreading range
    max_array = np.array([im_orig[:, :, 0].max(), im_orig[:, :, 1].max(), im_orig[:, :, 2].max()])
    min_array = np.array([im_orig[:, :, 0].min(), im_orig[:, :, 1].min(), im_orig[:, :, 2].min()])
    diff_array = np.array([0, 0, 0])
    for i in range(RGB_SIZE):
        diff_array[i] = max_array[i] - min_array[i]
    max_index = diff_array.argmax()
    max_dim = im_orig[:, :, max_index]
    interval_size = max_dim.size / n_quant
    hist_dim = np.histogram(max_dim, bins = NORMALIZE_CONST + 1, range = [0, NORMALIZE_CONST + 1])[0]
    cumulative = hist_dim.cumsum()
    z, q = np.array([0 for _ in range(n_quant + 1)]), [0 for _ in range(n_quant)]
    z[0], z[-1] = 0, NORMALIZE_CONST
    # creating the z array
    for i in range(1, n_quant):
        z[i] = np.where(cumulative > i * interval_size)[0][0]
    # creating the q array by selecting the max value of each [z[i]:z[i+1]] interval  middle half.
    for i in range(n_quant):
        first_index = int((z[i + 1] - z[i]) / 4)
        last_index = int(((z[i + 1] - z[i]) / 4) * 3)
        q[i] = cumulative[first_index:last_index].argmax()
    look_up = np.zeros(NORMALIZE_CONST + 1)
    for i in range(len(q)):
        look_up[z[i]:z[i + 1]] = q[i]
    quantized_max = look_up[im_orig.astype(int)]
    return quantized_max / NORMALIZE_CONST


img = read_image('gray.jpg', 1)
histogram_equalize(img * NORMALIZE_CONST)