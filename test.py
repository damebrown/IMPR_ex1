import os
import sol1
import matplotlib.pyplot as plt
import numpy as np
import sys
import skimage.color

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Enter the image files directory...')

    path = sys.argv[1]

    if not os.path.exists(path):
        raise IOError('Illegal path name: "{}"'.format(path))

    for directory, dirnames, filenames in os.walk(path):
        if not os.path.exists('{}_equalized'.format(path)):
            os.mkdir('{}_equalized'.format(path))

        if not os.path.exists('{}_quantized'.format(path)):
            os.mkdir('{}_quantized'.format(path))

        filenames = list(filter(lambda x: x.endswith('.jpg'), filenames))
        for representation in [1, 2]:
            for file in filenames:
                print('reading {}...'.format(file))
                im = sol1.read_image(os.path.join(directory,file), representation)
                print('equalizing {}...'.format(file))
                im_eq, hist, hist_eq = sol1.histogram_equalize(im)
                im_eq = np.clip(im_eq, 0, 1)
                cmap = 'viridis' if representation == 2 else 'gray'
                plt.imsave('{}_equalized/{}_{}_eq.jpg'.format(path, file.replace(".jpg", ""), representation), im_eq, cmap=cmap)

        for representation in [1, 2]:
            for file in filenames:
                print('reading {}...'.format(file))
                im = sol1.read_image(os.path.join(directory,file), representation)
                print('quantizing {}...'.format(file))
                for n_quant in [2, 3, 5, 50]:
                    im_quant, errors = sol1.quantize(im, n_quant,100)
                    im_quant = np.clip(im_quant, 0, 1)
                    cmap = 'viridis' if representation == 2 else 'gray'
                    plt.imsave('{}_quantized/{}_{}_{}_quant.jpg'.format(path, file.replace(".jpg", ""), representation, n_quant), im_quant, cmap=cmap)

                    for i in range(1, len(errors)):
                        if errors[i] > errors[i-1]:
                            print('errors are not monotonically decreasing at indices {}, {} for n_quant={}'.format(i-1, i, n_quant))
                            print('Errors: ',errors)
                            raise Exception('Test failed!')

        for file in filenames:
            print('reading {}...'.format(file))
            im = sol1.read_image(os.path.join(directory,file), 2)
            print('Testing RGB to YIQ...')
            np.testing.assert_almost_equal(sol1.rgb2yiq(im), skimage.color.rgb2yiq(im), decimal=3)
            print('RGB to YIQ ({}) equal to 3 decimals!'.format(file))

        for file in filenames:
            print('reading {}...'.format(file))
            im = sol1.read_image(os.path.join(directory,file), 2)
            print('Testing YIQ to RGB...')
            np.testing.assert_almost_equal(sol1.yiq2rgb(im), skimage.color.yiq2rgb(im), decimal=3)
            print('YIQ to RGB ({}) equal to 3 decimals!'.format(file))

    print('Test passed!')
