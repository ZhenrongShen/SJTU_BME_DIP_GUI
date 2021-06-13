# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.stats import entropy
from skimage import img_as_ubyte, morphology


def thresholdOTSU(image, nbins=256):
    assert image.ndim == 2, 'support 2D grayscale image only'
    hist, bin_edges = np.histogram(image.ravel(), nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
    # Calculate variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    
    return threshold


def thresholdMaxEntropy(image, nbins=256):
    assert image.ndim == 2, 'support 2D grayscale image only'
    hist, bin_edges = np.histogram(image.ravel(), nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    H = []
    for i in range(nbins-1):
        e = entropy(hist[0: i+1]) + entropy(hist[i+1:])
        H.append(e)
    idx = np.argmax(H)
    threshold = bin_centers[:-1][idx]
    
    return threshold


def conditional_dilation(binary_image):
    marker = morphology.opening(binary_image)
    print('Conditional Dilation Loop ...')
    while True:
        target = marker.copy()
        marker = morphology.dilation(marker)
        marker = marker * binary_image
        if np.array_equal(target, marker):
            print('Conditional Dilation Complete !')
            return marker


def grayscale_reconstruction(gray_image):
    marker = morphology.dilation(gray_image)
    eps = 1e-4
    print('Grayscale Reconstruction Loop ...')
    while True:
        target = marker.copy()
        marker = morphology.dilation(marker)
        marker = np.minimum(marker, gray_image)
        if np.max(np.abs(target - marker)) < eps:
            print('Grayscale Reconstruction Loop Complete !')
            return marker


def OBR_func(gray_image):
    open_image = morphology.opening(gray_image)
    reconst_image = grayscale_reconstruction(open_image)
    return reconst_image


def CBR_func(gray_image):
    close_image = morphology.closing(gray_image)
    reconst_image = grayscale_reconstruction(close_image)
    return reconst_image


def cond_dilation(img):
    pos = [[]]
    img_ = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_, (x, y), 3, (0, 0, 255), thickness=-1)
            cv2.imshow("marker", img_)
            pos[-1].append([x, y])
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img_, (x, y), 3, (0, 0, 255), thickness=-1)
            cv2.imshow("marker", img_)
            pos[-1].append([x, y])
            cv2.polylines(img_,[np.array(pos[-1], np.int32).reshape((-1,1,2))],True, (255, 0, 0))
            pos.append([])
    cv2.namedWindow("marker", cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("marker", on_EVENT_LBUTTONDOWN)
    cv2.imshow("marker", img_)
    while 1:
        cv2.imshow("marker", img_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    marker = np.zeros_like(img_)
    cv2.fillPoly(marker, np.array(pos[:-1]), (1,1,1))
    marker = marker[:,:,0]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tmp = np.zeros_like(marker)
    while 1:
        marker = cv2.dilate(marker, element)
        marker = cv2.bitwise_and(marker, img)
        if np.array_equal(marker, tmp):
            break
        tmp = marker.copy()
    return tmp