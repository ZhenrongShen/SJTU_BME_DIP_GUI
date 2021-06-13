# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy import ndimage
from skimage import io, img_as_ubyte, filters, morphology
import cv2

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import pyqtgraph as pg

from imageProcess import thresholdOTSU, thresholdMaxEntropy, OBR_func, CBR_func, conditional_dilation


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = loadUi('DIP_GUI.ui', self)
        
        ### Projcet 1
        self.ui.RawImageView_1.getHistogramWidget().setVisible(False)
        self.ui.RawImageView_1.ui.menuBtn.setVisible(False)
        self.ui.RawImageView_1.ui.roiBtn.setVisible(False)
        self.ui.GrayImageView_1.getHistogramWidget().setVisible(False)
        self.ui.GrayImageView_1.ui.menuBtn.setVisible(False)
        self.ui.GrayImageView_1.ui.roiBtn.setVisible(False)
        self.ui.BinaryImageView_1.getHistogramWidget().setVisible(False)
        self.ui.BinaryImageView_1.ui.menuBtn.setVisible(False)
        self.ui.BinaryImageView_1.ui.roiBtn.setVisible(False)

        self.ui.OpenButton_1.clicked.connect(self.openImage_1)
        self.ui.OTSUButton.clicked.connect(self.binary_otsu)
        self.ui.EntropyButton.clicked.connect(self.binary_entropy)
        self.ui.ThresholdSlider.valueChanged.connect(self.binary_thresh)
        self.ui.ThresholdSpin.valueChanged.connect(self.binary_thresh)

        ### Projcet 2
        self.ui.RawImageView_2.getHistogramWidget().setVisible(False)
        self.ui.RawImageView_2.ui.menuBtn.setVisible(False)
        self.ui.RawImageView_2.ui.roiBtn.setVisible(False)
        self.ui.GrayImageView_2.getHistogramWidget().setVisible(False)
        self.ui.GrayImageView_2.ui.menuBtn.setVisible(False)
        self.ui.GrayImageView_2.ui.roiBtn.setVisible(False)
        self.ui.EdgeImageView.getHistogramWidget().setVisible(False)
        self.ui.EdgeImageView.ui.menuBtn.setVisible(False)
        self.ui.EdgeImageView.ui.roiBtn.setVisible(False)
        self.ui.DenoisedImageView.getHistogramWidget().setVisible(False)
        self.ui.DenoisedImageView.ui.menuBtn.setVisible(False)
        self.ui.DenoisedImageView.ui.roiBtn.setVisible(False)

        self.ui.OpenButton_2.clicked.connect(self.openImage_2)
        self.ui.RobertsXButton.clicked.connect(self.run_roberts_x)
        self.ui.RobertsYButton.clicked.connect(self.run_roberts_y)
        self.ui.PrewittXButton.clicked.connect(self.run_prewitt_x)
        self.ui.PrewittYButton.clicked.connect(self.run_prewitt_y)
        self.ui.SobelXButton.clicked.connect(self.run_sobel_x)
        self.ui.SobelYButton.clicked.connect(self.run_sobel_y)
        self.ui.GaussianButton.clicked.connect(self.run_Gaussian)
        self.ui.MedianButton.clicked.connect(self.run_median)

        ### Projcet 3
        self.ui.RawImageView_3.getHistogramWidget().setVisible(False)
        self.ui.RawImageView_3.ui.menuBtn.setVisible(False)
        self.ui.RawImageView_3.ui.roiBtn.setVisible(False)
        self.ui.GrayImageView_3.getHistogramWidget().setVisible(False)
        self.ui.GrayImageView_3.ui.menuBtn.setVisible(False)
        self.ui.GrayImageView_3.ui.roiBtn.setVisible(False)
        self.ui.BinaryImageView_3.getHistogramWidget().setVisible(False)
        self.ui.BinaryImageView_3.ui.menuBtn.setVisible(False)
        self.ui.BinaryImageView_3.ui.roiBtn.setVisible(False)
        self.ui.MorphImageView_3.getHistogramWidget().setVisible(False)
        self.ui.MorphImageView_3.ui.menuBtn.setVisible(False)
        self.ui.MorphImageView_3.ui.roiBtn.setVisible(False)

        self.ui.OpenButton_3.clicked.connect(self.openImage_3)
        self.ui.BinaryDilationButton.clicked.connect(self.binary_dilation)
        self.ui.BinaryErosionButton.clicked.connect(self.binary_erosion)
        self.ui.BinaryOpeningButton.clicked.connect(self.binary_opening)
        self.ui.BinaryClosingButton.clicked.connect(self.binary_closing)

        ### Projcet 4
        self.ui.RawImageView_4.getHistogramWidget().setVisible(False)
        self.ui.RawImageView_4.ui.menuBtn.setVisible(False)
        self.ui.RawImageView_4.ui.roiBtn.setVisible(False)
        self.ui.GrayImageView_4.getHistogramWidget().setVisible(False)
        self.ui.GrayImageView_4.ui.menuBtn.setVisible(False)
        self.ui.GrayImageView_4.ui.roiBtn.setVisible(False)
        self.ui.BinaryImageView_4.getHistogramWidget().setVisible(False)
        self.ui.BinaryImageView_4.ui.menuBtn.setVisible(False)
        self.ui.BinaryImageView_4.ui.roiBtn.setVisible(False)
        self.ui.MorphImageView_4.getHistogramWidget().setVisible(False)
        self.ui.MorphImageView_4.ui.menuBtn.setVisible(False)
        self.ui.MorphImageView_4.ui.roiBtn.setVisible(False)

        self.ui.OpenButton_4.clicked.connect(self.openImage_4)
        self.ui.DistanceTransformButton.clicked.connect(self.distance_transform)
        self.ui.SkeletonButton.clicked.connect(self.skeletonize)
        self.ui.SkeletonRestorationButton.clicked.connect(self.skeleton_restore)

        ### Projcet 5
        self.ui.RawImageView_5.getHistogramWidget().setVisible(False)
        self.ui.RawImageView_5.ui.menuBtn.setVisible(False)
        self.ui.RawImageView_5.ui.roiBtn.setVisible(False)
        self.ui.GrayImageView_5.getHistogramWidget().setVisible(False)
        self.ui.GrayImageView_5.ui.menuBtn.setVisible(False)
        self.ui.GrayImageView_5.ui.roiBtn.setVisible(False)
        self.ui.MorphImageView_5.getHistogramWidget().setVisible(False)
        self.ui.MorphImageView_5.ui.menuBtn.setVisible(False)
        self.ui.MorphImageView_5.ui.roiBtn.setVisible(False)

        self.ui.OpenButton_5.clicked.connect(self.openImage_5)
        self.ui.GrayDilationButton.clicked.connect(self.gray_dilation)
        self.ui.GrayErosionButton.clicked.connect(self.gray_erosion)
        self.ui.GrayOpeningButton.clicked.connect(self.gray_opening)
        self.ui.GrayClosingButton.clicked.connect(self.gray_closing)

        ### Projcet 6
        self.ui.RawImageView_6.getHistogramWidget().setVisible(False)
        self.ui.RawImageView_6.ui.menuBtn.setVisible(False)
        self.ui.RawImageView_6.ui.roiBtn.setVisible(False)
        self.ui.GrayImageView_6.getHistogramWidget().setVisible(False)
        self.ui.GrayImageView_6.ui.menuBtn.setVisible(False)
        self.ui.GrayImageView_6.ui.roiBtn.setVisible(False)
        self.ui.BinaryImageView_6.getHistogramWidget().setVisible(False)
        self.ui.BinaryImageView_6.ui.menuBtn.setVisible(False)
        self.ui.BinaryImageView_6.ui.roiBtn.setVisible(False)
        self.ui.MorphImageView_6.getHistogramWidget().setVisible(False)
        self.ui.MorphImageView_6.ui.menuBtn.setVisible(False)
        self.ui.MorphImageView_6.ui.roiBtn.setVisible(False)

        self.ui.OpenButton_6.clicked.connect(self.openImage_6)
        self.ui.EdgeDetectionButton.clicked.connect(self.morph_edge_detect)
        self.ui.MorphGradientButton.clicked.connect(self.morph_gradient)
        self.ui.ReconstructionButton.clicked.connect(self.morph_reconst)


    #################################################################
    #                           Project 1                           #
    #################################################################
    def openImage_1(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "打开", os.getcwd(), 
                                                  "图片(*.jpg *.png *.bmp *.tif *.gif);;All Files(*)")
        if img_name == '':
            return 0
        else:
            # Load image
            self.raw_img1 = img_as_ubyte(io.imread(img_name))
            # Clear image views
            self.ui.RawImageView_1.clear()
            self.ui.GrayImageView_1.clear()
            self.ui.BinaryImageView_1.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img1.ndim==3 else (1, 0)
            self.ui.RawImageView_1.setImage(np.transpose(self.raw_img1, axis))
            # Display grayscale image
            self.gray_img1 = self.raw_img1 if self.raw_img1.ndim == 2 else cv2.cvtColor(self.raw_img1, cv2.COLOR_RGB2GRAY)
            self.ui.GrayImageView_1.setImage(np.transpose(self.gray_img1, (1,0)))
            # Enable binary thresholding operation
            self.ui.ThresholdSpin.setEnabled(True)
            self.ui.ThresholdSpin.setValue(0)
            self.ui.ThresholdSlider.setEnabled(True)
            self.ui.ThresholdSlider.setValue(0)
            # Display histogram
            self.ui.HistogramView.clear()
            hist, _ = np.histogram(self.gray_img1, bins=np.arange(257))
            self.ui.HistogramView.getPlotItem().plot(np.arange(257), hist, stepMode=True, fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))

    def binary_otsu(self):
        thr = thresholdOTSU(self.gray_img1)
        self.ui.ThresholdSlider.setValue(thr)
        self.ui.ThresholdSpin.setValue(thr)

    def binary_entropy(self):
        thr = thresholdMaxEntropy(self.gray_img1)
        self.ui.ThresholdSlider.setValue(thr)
        self.ui.ThresholdSpin.setValue(thr)

    def binary_thresh(self, threshold):
        try:
            self.ui.HistogramView.getPlotItem().removeItem(self.threshLine)
        except:
            pass
        binary_image = self.gray_img1 >= threshold
        self.ui.BinaryImageView_1.setImage(np.transpose(binary_image, (1,0)), levels=[0, 1])
        self.threshLine = self.ui.HistogramView.getPlotItem().addLine(x=threshold)

    #################################################################
    #                           Project 2                           #
    #################################################################
    def openImage_2(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "打开", os.getcwd(),
                                                  "图片(*.jpg *.png *.bmp *.tif *.gif);;All Files(*)")
        if img_name == '':
            return 0
        else:
            # Load image
            self.raw_img2 = img_as_ubyte(io.imread(img_name))
            # Clear image views
            self.ui.RawImageView_2.clear()
            self.ui.GrayImageView_2.clear()
            self.ui.EdgeImageView.clear()
            self.ui.DenoisedImageView.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img2.ndim == 3 else (1, 0)
            self.ui.RawImageView_2.setImage(np.transpose(self.raw_img2, axis))
            # Display grayscale image
            self.gray_img2 = self.raw_img2 if self.raw_img2.ndim == 2 else cv2.cvtColor(self.raw_img2, cv2.COLOR_RGB2GRAY)
            self.ui.GrayImageView_2.setImage(np.transpose(self.gray_img2, (1, 0)))

    def run_roberts_x(self):
        output_img = filters.roberts_pos_diag(self.gray_img2)
        self.ui.EdgeImageView.clear()
        self.ui.EdgeImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_roberts_y(self):
        output_img = filters.roberts_neg_diag(self.gray_img2)
        self.ui.EdgeImageView.clear()
        self.ui.EdgeImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_prewitt_x(self):
        output_img = filters.prewitt_v(self.gray_img2)
        self.ui.EdgeImageView.clear()
        self.ui.EdgeImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_prewitt_y(self):
        output_img = filters.prewitt_h(self.gray_img2)
        self.ui.EdgeImageView.clear()
        self.ui.EdgeImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_sobel_x(self):
        output_img = filters.sobel_v(self.gray_img2)
        self.ui.EdgeImageView.clear()
        self.ui.EdgeImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_sobel_y(self):
        output_img = filters.sobel_h(self.gray_img2)
        self.ui.EdgeImageView.clear()
        self.ui.EdgeImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_Gaussian(self):
        kernel_size = self.ui.KernelSpin_2.value()
        sigma = self.ui.SigmaSpin.value()
        output_img = cv2.GaussianBlur(self.gray_img2, (kernel_size, kernel_size), sigma)
        self.ui.DenoisedImageView.clear()
        self.ui.DenoisedImageView.setImage(np.transpose(output_img, (1, 0)))

    def run_median(self):
        kernel_size = self.ui.KernelSpin_2.value()
        kernel = np.ones((kernel_size, kernel_size))
        output_img = filters.median(self.gray_img2, selem=kernel)
        self.ui.DenoisedImageView.clear()
        self.ui.DenoisedImageView.setImage(np.transpose(output_img, (1, 0)))

    #################################################################
    #                           Project 3                           #
    #################################################################
    def openImage_3(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "打开", os.getcwd(),
                                                  "图片(*.jpg *.png *.bmp *.tif *.gif);;All Files(*)")
        if img_name == '':
            return 0
        else:
            # Load image
            self.raw_img3 = img_as_ubyte(io.imread(img_name))
            # Clear image views
            self.ui.RawImageView_3.clear()
            self.ui.GrayImageView_3.clear()
            self.ui.BinaryImageView_3.clear()
            self.ui.MorphImageView_3.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img3.ndim == 3 else (1, 0)
            self.ui.RawImageView_3.setImage(np.transpose(self.raw_img3, axis))
            # Display grayscale image
            self.gray_img3 = self.raw_img3 if self.raw_img3.ndim == 2 else cv2.cvtColor(self.raw_img3, cv2.COLOR_RGB2GRAY)
            self.ui.GrayImageView_3.setImage(np.transpose(self.gray_img3, (1, 0)))
            # Display binary image (using OTSU)
            thr = thresholdOTSU(self.gray_img3)
            self.binary_img3 = self.gray_img3 >= thr
            self.ui.BinaryImageView_3.setImage(np.transpose(self.binary_img3, (1, 0)), levels=[0, 1])

    def binary_dilation(self):
        kernel_size = self.ui.KernelSpin_3.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.dilation(self.binary_img3, selem=kernel)
        self.ui.MorphImageView_3.clear()
        self.ui.MorphImageView_3.setImage(np.transpose(output_img, (1, 0)))

    def binary_erosion(self):
        kernel_size = self.ui.KernelSpin_3.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.erosion(self.binary_img3, selem=kernel)
        self.ui.MorphImageView_3.clear()
        self.ui.MorphImageView_3.setImage(np.transpose(output_img, (1, 0)))

    def binary_opening(self):
        kernel_size = self.ui.KernelSpin_3.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.opening(self.binary_img3, selem=kernel)
        self.ui.MorphImageView_3.clear()
        self.ui.MorphImageView_3.setImage(np.transpose(output_img, (1, 0)))

    def binary_closing(self):
        kernel_size = self.ui.KernelSpin_3.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.closing(self.binary_img3, selem=kernel)
        self.ui.MorphImageView_3.clear()
        self.ui.MorphImageView_3.setImage(np.transpose(output_img, (1, 0)))

    #################################################################
    #                           Project 4                           #
    #################################################################
    def openImage_4(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "打开", os.getcwd(),
                                                  "图片(*.jpg *.png *.bmp *.tif *.gif);;All Files(*)")
        if img_name == '':
            return 0
        else:
            # Load image
            self.raw_img4 = img_as_ubyte(io.imread(img_name))
            # Clear image views
            self.ui.RawImageView_4.clear()
            self.ui.GrayImageView_4.clear()
            self.ui.BinaryImageView_4.clear()
            self.ui.MorphImageView_4.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img4.ndim == 3 else (1, 0)
            self.ui.RawImageView_4.setImage(np.transpose(self.raw_img4, axis))
            # Display grayscale image
            self.gray_img4 = self.raw_img4 if self.raw_img4.ndim == 2 else cv2.cvtColor(self.raw_img4, cv2.COLOR_RGB2GRAY)
            self.ui.GrayImageView_4.setImage(np.transpose(self.gray_img4, (1, 0)))
            # Display binary image (using OTSU)
            thr = thresholdOTSU(self.gray_img4)
            self.binary_img4 = self.gray_img4 >= thr
            self.ui.BinaryImageView_4.setImage(np.transpose(self.binary_img4, (1, 0)), levels=[0, 1])

    def distance_transform(self):
        if self.ui.EuclideanButton.isChecked():
            output_img = ndimage.distance_transform_edt(self.binary_img4)
        elif self.ui.CityBlockButton.isChecked():
            output_img = ndimage.distance_transform_cdt(self.binary_img4, metric='taxicab')
        elif self.ui.ChessboardButton.isChecked():
            output_img = ndimage.distance_transform_cdt(self.binary_img4, metric='chessboard')
        self.ui.MorphImageView_4.clear()
        self.ui.MorphImageView_4.setImage(np.transpose(output_img, (1, 0)))

    def skeletonize(self):
        output_img, dist = morphology.medial_axis(self.binary_img4, return_distance=True)
        self.skeleton = output_img
        self.dist = dist
        self.ui.MorphImageView_4.clear()
        self.ui.MorphImageView_4.setImage(np.transpose(output_img, (1, 0)))

    def skeleton_restore(self):
        assert self.skeleton is not None
        assert self.dist is not None
        output_img = morphology.reconstruction(self.skeleton, self.dist)
        self.ui.MorphImageView_4.clear()
        self.ui.MorphImageView_4.setImage(np.transpose(output_img, (1, 0)))

    #################################################################
    #                           Project 5                           #
    #################################################################
    def openImage_5(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "打开", os.getcwd(),
                                                  "图片(*.jpg *.png *.bmp *.tif *.gif);;All Files(*)")
        if img_name == '':
            return 0
        else:
            # Load image
            self.raw_img5 = img_as_ubyte(io.imread(img_name))
            # Clear image views
            self.ui.RawImageView_5.clear()
            self.ui.GrayImageView_5.clear()
            self.ui.MorphImageView_5.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img5.ndim == 3 else (1, 0)
            self.ui.RawImageView_5.setImage(np.transpose(self.raw_img5, axis))
            # Display grayscale image
            self.gray_img5 = self.raw_img5 if self.raw_img5.ndim == 2 else cv2.cvtColor(self.raw_img5, cv2.COLOR_RGB2GRAY)
            self.ui.GrayImageView_5.setImage(np.transpose(self.gray_img5, (1, 0)))

    def gray_dilation(self):
        kernel_size = self.ui.KernelSpin_5.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.dilation(self.gray_img5, selem=kernel)
        self.ui.MorphImageView_5.clear()
        self.ui.MorphImageView_5.setImage(np.transpose(output_img, (1, 0)))

    def gray_erosion(self):
        kernel_size = self.ui.KernelSpin_5.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.erosion(self.gray_img5, selem=kernel)
        self.ui.MorphImageView_5.clear()
        self.ui.MorphImageView_5.setImage(np.transpose(output_img, (1, 0)))

    def gray_opening(self):
        kernel_size = self.ui.KernelSpin_5.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.opening(self.gray_img5, selem=kernel)
        self.ui.MorphImageView_5.clear()
        self.ui.MorphImageView_5.setImage(np.transpose(output_img, (1, 0)))

    def gray_closing(self):
        kernel_size = self.ui.KernelSpin_5.value()
        # kernel = np.ones((kernel_size, kernel_size))
        kernel = morphology.disk(kernel_size)
        output_img = morphology.closing(self.gray_img5, selem=kernel)
        self.ui.MorphImageView_5.clear()
        self.ui.MorphImageView_5.setImage(np.transpose(output_img, (1, 0)))

    #################################################################
    #                           Project 6                           #
    #################################################################
    def openImage_6(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "打开", os.getcwd(),
                                                  "图片(*.jpg *.png *.bmp *.tif *.gif);;All Files(*)")
        if img_name == '':
            return 0
        else:
            # Load image
            self.raw_img6 = img_as_ubyte(io.imread(img_name))
            # Clear image views
            self.ui.RawImageView_6.clear()
            self.ui.GrayImageView_6.clear()
            self.ui.BinaryImageView_6.clear()
            self.ui.MorphImageView_6.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img6.ndim == 3 else (1, 0)
            self.ui.RawImageView_6.setImage(np.transpose(self.raw_img6, axis))
            # Display grayscale image
            self.gray_img6 = self.raw_img6 if self.raw_img6.ndim == 2 else cv2.cvtColor(self.raw_img6, cv2.COLOR_RGB2GRAY)
            self.ui.GrayImageView_6.setImage(np.transpose(self.gray_img6, (1, 0)))
            # Display binary image (using OTSU)
            thr = thresholdOTSU(self.gray_img6)
            self.binary_img6 = self.gray_img6 >= thr
            self.ui.BinaryImageView_6.setImage(np.transpose(self.binary_img6, (1, 0)), levels=[0, 1])

    def morph_edge_detect(self):
        if self.ui.StandardButton.isChecked():
            output_img = img_as_ubyte(morphology.dilation(self.binary_img6)) - \
                         img_as_ubyte(morphology.erosion(self.binary_img6))
        elif self.ui.ExternalButton.isChecked():
            output_img = img_as_ubyte(morphology.dilation(self.binary_img6)) - \
                         img_as_ubyte(self.binary_img6)
        elif self.ui.InternalButton.isChecked():
            output_img = img_as_ubyte(self.binary_img6) - \
                         img_as_ubyte(morphology.erosion(self.binary_img6))
        self.ui.MorphImageView_6.clear()
        self.ui.MorphImageView_6.setImage(np.transpose(output_img, (1, 0)))

    def morph_gradient(self):
        if self.ui.StandardButton.isChecked():
            output_img = (morphology.dilation(self.gray_img6) - morphology.erosion(self.gray_img6)) / 2
        elif self.ui.ExternalButton.isChecked():
            output_img = (morphology.dilation(self.gray_img6) - self.gray_img6) / 2
        elif self.ui.InternalButton.isChecked():
            output_img = (self.gray_img6 - morphology.erosion(self.gray_img6)) / 2
        self.ui.MorphImageView_6.clear()
        self.ui.MorphImageView_6.setImage(np.transpose(output_img, (1, 0)))

    def morph_reconst(self):
        if self.ui.ConditionalDilationButton.isChecked():
            output_img = conditional_dilation(self.binary_img6)
        elif self.ui.OBRButton.isChecked():
            output_img = OBR_func(self.gray_img6)
        elif self.ui.CBRButton.isChecked():
            output_img = CBR_func(self.gray_img6)
        self.ui.MorphImageView_6.clear()
        self.ui.MorphImageView_6.setImage(np.transpose(output_img, (1, 0)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
    