############################################################################## important libraries ##########################################################################
import streamlit as st
from PIL import Image as ima
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pandas as pd
import time

st.set_page_config(page_title="Sea Creatures 🌊", page_icon="🌊", layout="wide")

st.markdown("""
<div style="background-color: navy; width: 100%; padding: 30px; border-radius: 10px;">
   <h1 style="color: white; text-align: center;">Sea Creatures 🌊</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; font-family: Arial, sans-serif;">
   <h1 style="font-size: 40px; color: #333;">MARINE LIFE ENCYCLOPEDIA</h1>
   <p style="font-size: 18px; color: #0066cc;">
       Explore the Marine Life Encyclopedia to learn fun facts about marine animals.
   </p>
</div>
""", unsafe_allow_html=True)

##############################################################################################################################################################################################################
##################################################################  the load_model and predict functions are created by : Ali      ###########################################################################

def load_model(model_path):
   model = tf.keras.models.load_model(model_path)
   return model

def predict(image_path, model, class_names):
   img = cv2.imread(image_path)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = cv2.resize(img, (224, 224))
   img = img.astype(np.float32) / 255.0
   img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
   img = np.expand_dims(img, axis=0)
   predictions = model.predict(img)
   return class_names[np.argmax(predictions[0])]

def search_animal(data, prediction_result):
   prediction_result = prediction_result.strip().lower()
   matching_row = data[data['name'] == prediction_result]
   if not matching_row.empty:
       return {
           "Animal type": matching_row['animal_type'].iloc[0],
           "Scientific name": matching_row['Scientific name'].iloc[0],
           "Habitat": matching_row['Habitat'].iloc[0],
           "Physical Characteristics": matching_row['Physical Characteristics'].iloc[0],
           "Behavior": matching_row['Behavior'].iloc[0],
           "Fun Facts": matching_row['Fun Facts'].iloc[0],
           "Finally ": matching_row['end'].iloc[0]
       }
   return "No match found for the given prediction result."

with st.sidebar:
   st.title("Navigation")
   selected_page = st.radio("Go to:", ["Marine Life Encyclopedia", "Detection"])

########################################################################################################################################################################################
#                                                                                          Marine Life Encyclopedia
########################################################################################################################################################################################

if selected_page == "Marine Life Encyclopedia":
   # Load and display images
   image_directory = r"C:\Users\Gell G15\Desktop\samples"
   image_extensions = [".jpg", ".jpeg", ".png"]
   image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) 
                 if os.path.splitext(f)[1].lower() in image_extensions][:23]
   
   for i in range(0, len(image_paths), 4):
       cols = st.columns(4)
       for j, path in enumerate(image_paths[i:i + 4]):
           with cols[j]:
               try:
                   image = ima.open(path).resize((300, 300))
                   name = os.path.splitext(os.path.basename(path))[0]
                   st.image(image, use_container_width=True, caption=name)
                   
                   if st.button(f"See More about {name}", key=f"button_{i + j}"):
                       if name == 'Quiz':
                           st.markdown('<p style="color:black;font-weight:bold;">What animal has three hearts 🪸?</p>', 
                                     unsafe_allow_html=True)
                           time.sleep(5)
                           st.markdown('<p style="color:black;font-weight:bold;">The answer is Octopus 🐙</p>', 
                                     unsafe_allow_html=True)
                       else:
                           data = pd.read_csv(r"C:\Users\Gell G15\Desktop\sea animals test(1).csv")
                           data['name'] = data['name'].str.strip().str.lower()
                           info = search_animal(data, name)
                           for key, value in info.items():
                               st.markdown(f'<p style="color:black;font-weight:bold;">{key}</p>', 
                                         unsafe_allow_html=True)
                               st.markdown(f'<p style="color:black;">{value}</p>', 
                                         unsafe_allow_html=True)
               except Exception as e:
                   st.error(f"Error displaying image: {e}")
################################################################################################################################################################################
#                                                                                "Sea Creature Detection" 
################################################################################################################################################################################

elif selected_page == "Detection":
   st.title("Image Detection")
   uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
   
   if uploaded_file is not None:
       image = ima.open(uploaded_file)
       st.image(image, caption="Uploaded Image", use_container_width=True)
       
       class_names = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish',
                     'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers',
                     'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp',
                     'Squid', 'Starfish', 'Turtle', 'Whale']
       
       model_path = os.path.join(os.path.dirname(__file__), 'mobilenet_model3.h5')
       model = load_model(model_path)
       
       if st.button("Predict"):
           temp_path = "temp_image.jpg"
           with open(temp_path, "wb") as f:
               f.write(uploaded_file.getbuffer())
           
           predicted_class = predict(temp_path, model, class_names)
           st.markdown(f'<p style="color:black;font-weight:bold;font-size:30px;">Predicted Class: {predicted_class}</p>',
                      unsafe_allow_html=True)
           
           data = pd.read_csv(r"C:\Users\Gell G15\Desktop\sea animals test(1).csv")
           data['name'] = data['name'].str.strip().str.lower()
           info = search_animal(data, predicted_class)
           
           for key, value in info.items():
               st.markdown(f'<p style="color:black;font-weight:bold;">{key}</p>', 
                         unsafe_allow_html=True)
               st.markdown(f'<p style="color:black;">{value}</p>', 
                         unsafe_allow_html=True)
           
           os.remove(temp_path)