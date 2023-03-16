import numpy as np
import math
import cv2
from scipy.fftpack import dct, idct

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    
def embed_image(img, watermark, alpha):
    # Đọc ảnh gốc
    I0 = img
    cv2.imshow('Anh goc',I0) 

    # Đọc ảnh QR
    QR = watermark
    cv2.imshow('Anh QR goc',cv2.resize(QR, [300, 300])) 
    QR = cv2.resize(QR, [100, 100])

    # chuyển ảnh gốc sang kênh màu đỏ
    I1 = I0[:,:,2]

    # Thực hiện biến đổi DCT
    Idct = dct2(I1)

    # Thêm QR vào ảnh
    Idct[650:750, 450:550] =  QR[0:100, 0:100,2] * alpha
    cv2.imshow("Anh sau khi bien doi DCT voi ma QR o kenh mau do",Idct)

    # Thực hiện biến đổi ngược DCT
    I12 = idct2(Idct) 
    
    # Thay đổi kênh màu đỏ của ảnh gốc bằng ảnh đã được nhúng mã QR
    Ih = I0
    Ih[:,:,2] = I12[:,:]
    
    return Ih

def extract_image(embed_img):
    # Thực hiện biến đổi DCT với ảnh đã được nhúng trên kênh màu đỏ
    Idct2 = dct2(embed_img[:,:,2])
    cv2.imshow('Anh sau khi bien doi DCT de lay ma nhung ma QR o kenh mau do',Idct2)

    # Tạo một mảng 100x100 để lưu trữ ảnh QR
    Qr = np.zeros((100,100))

    # Lấy ảnh QR từ vị trí ảnh đã được nhúng
    Qr[0:100, 0:100] =  Idct2[650:750, 450:550] 
    
    return Qr

if __name__ == "__main__":
    # Đọc ảnh đầu vào
    img = cv2.imread('13.jpg')
    watermark = cv2.imread('QRcode.png')

    # Nhúng ảnh 
    embedded_img = embed_image(img, watermark, 245)
    cv2.imwrite('DCTembedded_img.png', embedded_img)
    cv2.imshow('Anh sau khi duoc nhung ma QR o kenh mau do', cv2.imread('DCTembedded_img.png'))
    
    # Trích xuất QR code 
    extracted_img = extract_image(embedded_img) 
    cv2.imwrite('DCTextracted_watermark.png', np.uint8(cv2.resize(extracted_img, (300, 300))))
    cv2.imshow("Anh sau khi xuat thanh cong ", cv2.imread('DCTextracted_watermark.png'))

    cv2.waitKey(0)
    cv2.destroyAllWindows()