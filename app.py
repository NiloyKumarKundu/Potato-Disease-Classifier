import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st


MODEL = tf.keras.models.load_model('saved_models\\potatoes.h5')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


st.title("Potato Disease Classifier")


image = st.file_uploader('Upload your image here', type=['png', 'jpg', 'jpeg'])

if image is not None:
    input_image = Image.open(image)


    with st.spinner('AI is at work...'):
        np_image = np.array(input_image)
        img_batch = np.expand_dims(np_image, 0)

        st.image(input_image)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confindence = np.max(predictions[0])

        st.write(f'Predicted Class: {predicted_class}')
        confindence_score = confindence * 100.0
        st.write(f'Accuracy: {confindence_score}%')

else:
    st.write('Please upload an image')

st.markdown('Github link: [Click here!](https://github.com/NiloyKumarKundu/Potato-Disease-Classification)')