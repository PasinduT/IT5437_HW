import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

st.title('Question 4: Intensity Transformations')
st.set_page_config(page_title="Question 4", layout="wide")

with st.sidebar:
    a = st.slider('Select a', 0.0, 1.0, 0.4)
    sigma = st.slider('Select sigma', 1, 100, 70)

def get_intensity_transform(a, sigma):
    def intensity_transform(x):
        return min((int(x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2)))), 255)
    return intensity_transform

@st.cache_data
def load_image():
    img = Image.open('a1images/spider.png')
    img = np.array(img)
    return img

img = load_image()  

f = get_intensity_transform(a, sigma)
transform = np.array([f(i) for i in range(256)], dtype=np.uint8)
st.line_chart(transform, height=200, use_container_width=False, width=200)
s1, s2 = st.columns(2)

with s1:
    st.image(img, caption='Original Image', use_container_width=True)

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img_hsv[:, :, 1] = cv.LUT(img_hsv[:, :, 1], transform)

img_transformed = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)

with s2:
    st.image(img_transformed, caption='Transformed Image', use_container_width=True)