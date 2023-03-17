import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def embed_image(img, watermark, alpha):
    cv2.imshow('Anh goc',img)
    cv2.imshow('QR goc goc',cv2.resize(watermark, (300, 300)))
    
    # Resize ảnh được nhúng bằng kích thước ảnh gốc
    watermark = cv2.resize(watermark, (img.shape[1], img.shape[0]))
    
    # Biến đổi DWT
    coeffs_img = pywt.dwt2(img, 'haar')
    coeffs_wm = pywt.dwt2(watermark, 'haar')
    
    cA_img, (cH_img, cV_img, cD_img) = coeffs_img
    cA_wm, (cH_wm, cV_wm, cD_wm) = coeffs_wm

    # Nhúng watermark vào ảnh
    cA_embed = cA_img + alpha * cA_wm
    
    # Biên đổi ngược DWT để nhận được ảnh nhúng watermark
    embed_img = pywt.idwt2((cA_embed, (cH_img, cV_img, cD_img)), 'haar')
    
    return embed_img

def extract_image(embed_img, original_img, alpha):
    # Biến đổi DWT vói ảnh đã được nhúng và ảnh gốc
    coeffs_embed = pywt.dwt2(embed_img, 'haar')
    coeffs_orig = pywt.dwt2(original_img, 'haar')
    
    cA_embed, (cH_embed, cV_embed, cD_embed) = coeffs_embed
    cA_orig, (cH_orig, cV_orig, cD_orig) = coeffs_orig

    # trích xuất ảnh
    cA_extract = (cA_embed - cA_orig) / alpha

    # Biên đổi ngược DWT để nhận được ảnh trích xuất
    extracted_img = pywt.idwt2((cA_extract, (cH_orig, cV_orig, cD_orig)), 'haar')
    
    # Resize ảnh sau khi lấy được kết quả
    extracted_img = cv2.resize(extracted_img, (300, 300))
    
    return extracted_img

if __name__ == "__main__":
    # Đọc ảnh đầu vào
    img = cv2.imread('13.jpg')
    watermark = cv2.imread('QRcode.png')
    
    # Khóa alpha 
    alpha = 0.02

    # Nhúng ảnh 
    embedded_img = embed_image(img, watermark, alpha)
    
    # lưu ảnh nhúng và hiển thị ở dạng uint8
    cv2.imwrite('DWTembedded_img.png', embedded_img)
    cv2.imshow('Embedded Image', cv2.imread('DWTembedded_img.png'))
    
    # Trích xuất QR code 
    extracted_img = extract_image(embedded_img, img, alpha)
    
    # lưu ảnh trích xuất và hiển thị ở dạng uint8
    cv2.imwrite('DWTextracted_watermark.png', extracted_img)
    cv2.imshow('Extracted Image', cv2.imread('DWTextracted_watermark.png'))
    

cv2.waitKey(0)
cv2.destroyAllWindows()
