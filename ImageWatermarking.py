import numpy as np
import cv2
import pywt

def DWT(coverImage, waterMarkImage):
    # Đọc và thay đổi kích thước ảnh
    coverImage = cv2.resize(coverImage, (300,300))
    cv2.imshow('coverImage', coverImage)
    waterMarkImage = cv2.resize(waterMarkImage, (150,150))
    cv2.imshow('waterMarkImage', waterMarkImage)
    
    # Thực hiện DWT trên ảnh cover
    coverImage = np.float32(coverImage)
    coverImage /= 255
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC
    
    waterMarkImage = np.float32(waterMarkImage)
    waterMarkImage /= 255
    
    # Nhúng 
    apha_c = 1
    apha_w = 0.07
    coeffW = (apha_c * cA + apha_w * waterMarkImage, (cH, cV, cD))
    waterMarkImage = pywt.idwt2(coeffW, 'haar')
    cv2.imshow('waterMarkImage', waterMarkImage)
    
    # Rút trích 
    coeffWM = pywt.dwt2(waterMarkImage, 'haar')
    hA, (hH, hV, hD) = coeffWM
    
    extracted = (hA - apha_c * cA) / apha_w
    extracted *= 255 
    extracted = np.uint8(extracted)
    cv2.imshow('extracted', extracted)
    
coverImage = cv2.imread('image.png',0)
watermarkImage = cv2.imread('QRCode.png',0)
DWT(coverImage, watermarkImage)
cv2.waitKey(0)
cv2.destroyAllWindows()