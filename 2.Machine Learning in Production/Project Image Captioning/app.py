import streamlit as st 
from PIL import Image
import predict 
import numpy as np

from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

#app title
st.title('CapBot')

#uploaded_image

file = st.file_uploader("Choose and image ...",type="jpg")
temp_file = NamedTemporaryFile(delete=False)


if file is not None:
	image = Image.open(file)
	st.image(image,caption='Uploaded Image',width=100)
	st.write("")
	
	if st.button("generate captions"):
		st.write("CapBot generates your caption")
		image = Image.open(file)
		
		caption = predict.get_prediction(image)
		st.markdown(caption)
	
