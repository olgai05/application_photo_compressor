import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage import io

# Name of application 
st.title("Photo Compressor Application")
st.write(" Here you can covert your picture to the picture with low resolution")

# Example 
url = "https://thumbs.dreamstime.com/b/lone-sailboat-17336655.jpg"
image = io.imread(url)[:,:,0]


#SVD

def svd(image_test):
    U, sing_values, V = np.linalg.svd(image_test)
    sigma = np.zeros(shape=image_test.shape)
    np.fill_diagonal(sigma, sing_values)
    top_k = int(sigma.shape[0]/10)
    tranc_U = U[:, :top_k]
    tranc_sigma = sigma[:top_k, :top_k]
    tranc_V = V[:top_k, :]
    svd_image = tranc_U@tranc_sigma@tranc_V
    return svd_image
image1 = svd(image)

#U, sing_values, V = np.linalg.svd(image)
#sigma = np.zeros(shape=image.shape)
#np.fill_diagonal(sigma, sing_values)
#top_k = int(sigma.shape[0]/10)
#tranc_U = U[:, :top_k]
#tranc_sigma = sigma[:top_k, :top_k]
#tranc_V = V[:top_k, :]
fig1, ax = plt.subplots(1,2, figsize =(6, 12))
ax[0].imshow(image, cmap = "gray")
ax[0].axis('off')
ax[1].imshow(image1, cmap="grey")
ax[1].axis("off")
st.pyplot(fig1)
# 
# Download photos files 
uploaded_file = st.file_uploader("Download your picture", type=['png','jpg'])
if uploaded_file is not None:
    image2 = io.imread(uploaded_file)[:,:,0]

    #SVD
    image3 = svd(image2)
    fig2, ax = plt.subplots(1,2, figsize =(6, 12))
    ax[0].imshow(image2, cmap = "gray")
    ax[0].axis('off')
    ax[1].imshow(image3, cmap="grey")
    ax[1].axis("off")
    st.pyplot(fig2)




# Selecting the singular number 

# Calclation svd

# dempnstration results and initial picture
