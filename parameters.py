import numpy as np
import cv2
import math
from fileLogger import FileLogger
import sys


def pixels_to_mm(pixels, dpi, scale):
    """
    Переводит размер в пикселях в миллиметры
    :param pixels: размер в пикселях
    :param dpi: DPI
    :param scale: увеличение микроскопа
    :return: размер в миллиметрах
    """
    MILLIMETERS_IN_INCH = 25.4
    return MILLIMETERS_IN_INCH * pixels / dpi / scale


def pixels_to_square_mm(pixels, dpi, scale):
    """
    Переводит площадь в пикселях в площадь в кв.мм.
    :param pixels: площадь в пикселях
    :param dpi: DPI
    :param scale: увеличение микроскопа
    :return: площадь в квадратных миллиметрах
    """
    square_of_one_pixel = pixels_to_mm(1 ,dpi=dpi, scale=scale) ** 2

    return square_of_one_pixel * pixels

if __name__ == '__main__':

    DPI = 96 # dpi в пикселях/дюйм
    SCALE = 100 # увеличение микроскопа
    logger = FileLogger()

    # load the image
    if len(sys.argv) < 2:
        image = cv2.imread("image.png")
    else:
        image = cv2.imread(sys.argv[1])

    mask = cv2.inRange(image, np.array([0, 0, 0], dtype="uint8"), np.array([100, 100, 100], dtype="uint8"))

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    # CALCULATE PARAMETERS
    kernelCount = len(contours)
    logger.log("Число зерен", str(kernelCount))

    cntSquares = [pixels_to_square_mm(cv2.contourArea(cnt), DPI, SCALE) for cnt in contours]

    minS = min(cntSquares)
    maxS = max(cntSquares)
    averS = sum(cntSquares) / len(contours)
    logger.log("Минимальная площадь зерна, кв. мм.", str(minS))
    logger.log("Максимальная площадь зерна, кв. мм.", str(maxS))
    logger.log("Средняя площадь зерна, кв. мм.", str(averS))

    height, width, _ = image.shape
    image_square = pixels_to_mm(height, DPI, SCALE) * pixels_to_mm(width, DPI, SCALE)
    kernels_on_1_mm = kernelCount / image_square
    logger.log("Число зерен на площади 1 кв. мм.", str(kernels_on_1_mm))

    cntPerimeters = [pixels_to_mm(cv2.arcLength(cnt, True), DPI, SCALE) for cnt in contours]

    minP = min(cntPerimeters)
    maxP = max(cntPerimeters)
    averP = sum(cntPerimeters) / len(contours)
    logger.log("Минимальный периметр зерна, мм", str(minP))
    logger.log("Максимальный периметр зерна, мм", str(maxP))
    logger.log("Средний периметр зерна, мм", str(averP))

    # Балл зерна, метод подсчета зерен, ГОСТ-5639-82, 3.4
    ball = round(math.log(averS)/math.log(0.5) - 3)
    logger.log("Балл зерна", str(ball))

    # Show
    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
