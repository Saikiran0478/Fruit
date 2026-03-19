import numpy as np
from skimage import color
from skimage.feature import graycomatrix, graycoprops
import scipy.stats as stats

def channels_first_transform(image):
    return image.transpose((2, 0, 1))

def remove_green_pixels(image):
    channels_first = channels_first_transform(image)
    r, g, b = channels_first
    mask = np.multiply(g > r, g > b)
    channels_first = np.multiply(channels_first, mask)
    return channels_first.transpose(1, 2, 0)

def rgb2gray(image):
    return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0]):
    single_channel = image if len(image.shape) == 2 else rgb2gray(image)
    return graycomatrix(single_channel, offsets, angles)

def glcm_features(glcm_matrix):
    return np.array([
        graycoprops(glcm_matrix, 'correlation'),
        graycoprops(glcm_matrix, 'contrast'),
        graycoprops(glcm_matrix, 'energy'),
        graycoprops(glcm_matrix, 'homogeneity'),
        graycoprops(glcm_matrix, 'dissimilarity'),
    ]).flatten()

def histogram_features_bucket_count(image):
    image = channels_first_transform(image).reshape(3, -1)
    r, g, b = image

    r_hist = np.histogram(r, bins=26, range=(0,255))[0]
    g_hist = np.histogram(g, bins=26, range=(0,255))[0]
    b_hist = np.histogram(b, bins=26, range=(0,255))[0]

    return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
    hist = np.histogram(image.flatten(), bins=255, range=(0,255))[0]

    return np.array([
        np.mean(hist),
        np.std(hist),
        stats.entropy(hist),
        stats.kurtosis(hist),
        stats.skew(hist),
        np.sqrt(np.mean(np.square(hist)))
    ])

def texture_features(image, offsets=[1], angles=[0], remove_green=True):
    if remove_green:
        image = remove_green_pixels(image)
    gray = rgb2gray(image)
    glcmatrix = glcm(gray, offsets, angles)
    return glcm_features(glcmatrix)

def extract_features(image):
    offsets = [1,3,10,20]
    angles = [0, np.pi/4, np.pi/2]

    ch = channels_first_transform(image)

    return np.concatenate((
        texture_features(image, offsets, angles),
        texture_features(image, offsets, angles, False),
        histogram_features_bucket_count(image),
        histogram_features(ch[0]),
        histogram_features(ch[1]),
        histogram_features(ch[2]),
    ))
