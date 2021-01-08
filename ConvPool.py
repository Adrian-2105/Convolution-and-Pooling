'''
Fuentes:
- Operación convolucional (explicado con bastante detalle): https://prvnk10.medium.com/the-convolution-operation-48d72a382f5a
- Implementación de la operación convolucional: https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
- Pooling: https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
'''

import cv2
import numpy as np
import itertools as it

# Macros
RED = 2
GREEN = 1
BLUE = 0

'''
Apply a pooling function based on a value to a 2D image

:param image: 2D Image objective
:param n: int pooling value, must be a divisor of the height and width of the array
:param pool_func: Pooling function to be applied. It must be a function of the numpy library 
                like (f.e) np.sum(), np.max(), np.min(), np.average(), ...

:returns: Pooled 2D image
'''
def pooling2d(image, n, pool_func):
    # Get the image dimensions
    hImg, wImg = image.shape[0], image.shape[1]

    # Check if the n value is valid
    if hImg % n != 0 or wImg % n != 0:
        print('n is an invalid parameter')
        return

    # Create an empty matrix
    newImg = np.zeros([hImg // n, wImg // n])

    # Calculation of the resulting image
    for i, j in it.product(range(0, hImg, n), range(0, wImg, n)):
        newImg[i // n, j // n] = pool_func(image[i:i + n, j:j + n])

    return newImg

'''
Apply a pooling function based on a value to a 3D image

:param image: 3D Image objective
:param n: int pooling value, must be a divisor of the height and width of the array
:param pool_func: Pooling function to be applied. It must be a function of the numpy library 
                like (f.e) np.sum(), np.max(), np.min(), np.average(), ...

:returns: Pooled 3D image
'''
def pooling3d(image, n, pool_func):
    # Get the image dimensions
    hImg, wImg, dImg = image.shape[0], image.shape[1], image.shape[2]

    # Check if the n value is valid
    if hImg % n != 0 or wImg % n != 0:
        print('n is an invalid parameter')
        return

    # Create an empty matrix
    newImg = np.zeros([hImg // n, wImg // n, dImg])

    # Calculation of the resulting image
    for i, j, k in it.product(range(0, hImg, n), range(0, wImg, n), range(0, dImg)):
        newImg[i // n, j // n, k] = pool_func(image[i:i + n, j:j + n, k])

    return newImg


'''
Apply a convolution to a 2D image

:param image: 2D Image objective
:param filter: 2D Matrix with the filter to be applied to the image
:param padding: int value with the fill value to be applied to the image
:param stride: int value with filter stride for the image. Reduces the size of the resulting image

:returns: 2D Image with convolution applied
'''
def convolution2d(image, filter, padding=0, stride=1):
    padding = int(padding)
    stride = int(stride)

    # Get the image dimensions
    hImg, wImg = image.shape[0], image.shape[1]

    # Get the filters dimensions
    hFil, wFil = filter.shape[0], filter.shape[1]

    # Create an empty matrix
    hNewImg, wNewImg = (hImg - hFil + 2 * padding) // stride + 1, (wImg - wFil + 2 * padding) // stride + 1
    newImg = np.zeros([hNewImg, wNewImg])

    # Apply the padding to both sides
    if padding != 0:
        imagePadded = np.zeros([hImg + 2 * padding, wImg + 2 * padding])
        imagePadded[padding:-1 * padding, padding:-1 * padding] = image
    else:
        imagePadded = image

    # Calculation of the resulting image applying the filters
    for i, j in it.product(range(0, hNewImg), range(0, wNewImg)):
        newImg[i, j] = np.sum(filter * imagePadded[i * stride:i * stride + hFil, j * stride:j * stride + wFil])

    return newImg

'''
Apply a convolution to a 3D image

:param image: 3D Image objective
:param filters: 3D Matrix with the filters to be applied to the image
:param padding: int value with the fill value to be applied to the image
:param stride: int value with filters stride for the image. Reduces the size of the resulting image

:returns: 3D Image with convolution applied
'''
def convolution3d(image, filters, padding=0, stride=1):
    padding = int(padding)
    stride = int(stride)

    # Get the image dimensions
    hImg, wImg, dImg = image.shape[0], image.shape[1], image.shape[2]

    # Get the filters dimensions
    nFil, hFil, wFil, dFil = filters.shape[0], filters.shape[1], filters.shape[2], filters.shape[3]

    # Check parameters
    if dImg != dFil:
        print('Invalid filters')
        return

    # Create an empty matrix
    hNewImg, wNewImg, dNewImg = (hImg - hFil + 2 * padding) // stride + 1, (wImg - wFil + 2 * padding) // stride + 1, nFil
    newImg = np.zeros([hNewImg, wNewImg, dNewImg])

    # Apply the padding to all sides
    if padding != 0:
        imagePadded = np.zeros([hImg + 2 * padding, wImg + 2 * padding, dImg])
        imagePadded[padding:-1 * padding, padding:-1 * padding] = image
    else:
        imagePadded = image

    # Calculation of the resulting image applying the filters
    for m, i, j in it.product(range(0, dNewImg), range(0, hNewImg), range(0, wNewImg), ):
        newImg[i, j, m] = np.sum(filters[m] * imagePadded[i * stride:i * stride + hFil, j * stride:j * stride + wFil]) // dNewImg

    return newImg






'''
Loads an image as a matrix of grayscale values

:param filename: File name of the image
'''
def load2dImage(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

'''
Loads an image as a 3D matrix of RGB values

:param image: File name of the image
'''
def load3dImage(filename):
    return cv2.imread(filename)

'''
Save a cv2 image for the specified name

:param filename: File name of the image to be saved
:param image: cv2 image
'''
def saveImage(filename, image):
    cv2.imwrite(filename, image)

def testFilters():
    # List of filters for a 2D example
    edge2dFilter = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

    sharpered2dFilter = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    # List of filters for a 3D example
    sharpered3dFilter = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]],
                                  [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]],
                                  [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
    sharpered3dFilterGreen = np.array([sharpered3dFilter-1, sharpered3dFilter, sharpered3dFilter-1])

    edge3dFilter = np.array([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                             [[-1, -1, -1], [8, 8, 8], [-1, -1, -1]],
                             [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]])
    edge3dFilterRed = np.array([edge3dFilter-1, edge3dFilter-1, edge3dFilter])

    # Opening the image file
    filename = input('Enter the name of an image (found in the same directory): ')
    try:
        image2d = load2dImage(filename)
        image3d = load3dImage(filename)
    except:
        print('The file', filename, 'cannot be found or cannot be read')
        exit(1)

    # Image processing
    print('\nProcessing image, it will take a while...\n')
    ok = 0
    # 2D
    try:
        saveImage('2dSharpered10Pad.jpg', convolution2d(image2d, sharpered2dFilter, 10))
        print('Generated 2dSharpered10Pad.jpg')
        ok += 1
    except:
        print('Error with 2dSharpered10Pad.jpg')
    try:
        saveImage('2dEdge2Stride.jpg', convolution2d(image2d, edge2dFilter, 0, 2))
        print('Generated 2dEdge2Stride.jpg')
        ok += 1
    except:
        print('Error with 2dEdge2Stride.jpg')
    try:
        saveImage('2d4Pooled.jpg', pooling2d(image2d, 4, np.average))
        print('Generated 2d4Pooled.jpg')
        ok += 1
    except:
        print('Error with 2d4Pooled.jpg')

    # 3D
    try:
        saveImage('3dSharpered2Stride.jpg', convolution3d(image3d, sharpered3dFilterGreen, 0, 2))
        print('Generated 3dSharpered2Stride.jpg')
        ok += 1
    except:
        print('Error with 3dSharpered2Stride.jpg')
    try:
        saveImage('3dEdgeRed.jpg', convolution3d(image3d, edge3dFilterRed))
        print('Generated 3dEdgeRed.jpg')
        ok += 1
    except:
        print('Error with 3dEdgeRed.jpg')
    try:
        saveImage('3d2Pooled.jpg', pooling3d(image3d, 2, np.average))
        print('Generated 3d2Pooled.jpg')
        ok += 1
    except:
        print('Error with 3d2Pooled.jpg')

    print()
    print(ok, '/6 images generated\n')
    print('Done!')
