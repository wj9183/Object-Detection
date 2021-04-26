import streamlit as st
import cv2
import numpy as np



#이미지 프로세싱하는 함수. 변수로 이미지를 받음.
# def process_image(img):
#   """ 이미지 리사이즈하고, 차원 확장
#   img : 원본 이미지
#   결과는 (64, 64, 3)으로 프로세싱된 이미지 반환 """

#   image_org = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
#   image_org = np.array(image_org, dtype = 'float32')
#   image_org = image_org / 255.0
#   image_org = np.expand_dims(image_org, axis = 0)

#   return image_org


def yolo():
    pass