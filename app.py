import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import  Image

st.title('Cat Dog Classifier')

#Step 1 : Load the model
model=load_model('cats_dogs_small_3.h5')


# Step2: Load the image to predict
file_uploaded=st.file_uploader('Choose the database', accept_multiple_files=False)
if file_uploaded is not None:
    file_name=file_uploaded
else:
    file_name='image.jpg'
    
if st.checkbox('View Image', False):
    image=Image.open(file_name)
    st.image(image)
    
#Step3: Preprocess the image
img=load_img(file_name, target_size=(150, 150))
img_array=img_to_array(img)
img_array_final=np.expand_dims(img_array, axis=0)

# Step4: Predict the image and print the result
prediction=int(model.predict(img_array_final)[0][0])
if st.button("PREDICT"):
    if prediction==1:
        st.subheader("The image is a Dog")
    else:
        st.subheader("The image is a Cat")
