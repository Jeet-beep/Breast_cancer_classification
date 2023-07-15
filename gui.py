import pickle
import streamlit as st
import numpy as np


intro="*Breast cancer is a significant global health concern affecting millions of women and, in rare cases, men. It is the most commonly diagnosed cancer and a leading cause of cancer-related deaths among women. However, with advancements in medical research and technology, early detection has proven to be crucial in improving survival rates and treatment outcomes.To address this critical need, our project aims to develop a breast cancer prediction model that utilizes advanced data analysis techniques and machine learning algorithms. By harnessing the power of artificial intelligence, we aim to empower individuals and healthcare professionals with an accurate and reliable tool for assessing an individual's risk of developing breast cancer.The primary goal of our project is to provide an accessible and user-friendly platform that can potentially identify high-risk individuals early on. By identifying individuals at an increased risk, we can facilitate timely screenings, promote preventive measures, and ultimately contribute to reducing the burden of breast cancer.*"


st.markdown("<h1 style='text-align: center; color: Black;'>Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.header(':black[*please insert values in proper format*]')
st.sidebar.title('Introduction :female-doctor:')
st.sidebar.markdown(intro)



import streamlit as st
from PIL import Image

image = Image.open('doctor-clinic-illustration_1270-69.png')
image=image.resize((200,200))

st.image(image, caption='Doctor')

model_name="breast_svm.pkl"
with open(model_name,'rb') as file:
  a=pickle.load(file)
print(a)

radius_mean = st.text_input( "**radius_mean**",placeholder="87623XXX",key='1')
texture_mean = st.text_input( "**texture_mean**",placeholder="87623XXX",key='2')
perimeter_mean = st.text_input( "**perimeter_mean**",placeholder="87623XXX",key='3')
area_mean = st.text_input( "**area_mean**",placeholder="87623XXX",key='4')
smoothness_mean = st.text_input( "**smoothness_mean**",placeholder="87623XXX",key='5')
compactness_mean = st.text_input( "**compactness_mean**",placeholder="87623XXX",key='6')
concavity_mean = st.text_input( "**concavity_mean**",placeholder="87623XXX",key='7')
concave_points_mean = st.text_input( "**concave points_mean**",placeholder="87623XXX",key='8')
symmetry_mean = st.text_input( "**symmetry_mean'**",placeholder="87623XXX",key='9')
fractal_dimension_mean= st.text_input( "**fractal_dimension_mean**",placeholder="87623XXX",key='10')
radius_se = st.text_input( "**radius_se**",placeholder="87623XXX",key='11')
texture_se = st.text_input( "**texture_se**",placeholder="87623XXX",key='12')
perimeter_se = st.text_input( "**perimeter_se**",placeholder="87623XXX",key='13')
area_se = st.text_input( "**area_se**",placeholder="87623XXX",key='14')
smoothness_se = st.text_input( "**smoothness_se**",placeholder="87623XXX",key='15')
compactness_se = st.text_input( "**compactness_se**",placeholder="87623XXX",key='16')
concavity_se = st.text_input( "**concavity_se**",placeholder="87623XXX",key='17')
concave_points_se=st.text_input("**concave_points_se**",placeholder="3842579384523",key='18')
symmetry_se = st.text_input( "**symmetry_se**",placeholder="87623XXX",key='19')
fractal_dimension_se = st.text_input( "**fractal_dimension_se**",placeholder="87623XXX",key='20')
radius_worst = st.text_input( "**radius_worst**",placeholder="87623XXX",key='21')
texture_worst = st.text_input( "**texture_worst**",placeholder="87623XXX",key='22')
perimeter_worst = st.text_input( "**perimeter_worst**",placeholder="87623XXX",key='23')
area_worst = st.text_input( "**area_worst**",placeholder="87623XXX",key='24')
smoothness_worst= st.text_input( "**smoothness_worst**",placeholder="87623XXX",key='25')
compactness_worst = st.text_input( "**compactness_worst**",placeholder="87623XXX",key='26')
concavity_worst = st.text_input( "**concavity_worst**",placeholder="87623XXX",key='27')
concave_points_worst = st.text_input( "**concave points_worst**",placeholder="87623XXX",key='28')
symmetry_worst = st.text_input( "**symmetry_worst**",placeholder="87623XXX",key='29')
fractal_dimension_worst= st.text_input( "**fractal_dimension_worst**",placeholder="87623XXX",key='30')


def ref():
    input_data=(radius_mean, texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se, texture_se, perimeter_se, area_se, smoothness_se,compactness_se, concavity_se, concave_points_se,symmetry_se,fractal_dimension_se, radius_worst, texture_worst,
                perimeter_worst, area_worst, smoothness_worst,
                    compactness_worst, concavity_worst, concave_points_worst,
                    symmetry_worst, fractal_dimension_worst)
   
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data) # converting this list into numpy array

    # reshape the numpy array as we are predicting for one instance

    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
    print(input_data_reshaped)

    st.empty()

    predictions = a.predict(input_data_reshaped)
    print(predictions)
    if predictions[0] == 1:
        st.write("M")
    else:
        st.write("B")


st.button("Submit",on_click=ref)









# input_data=(id, radius_mean, texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se, texture_se, perimeter_se, area_se, smoothness_se,compactness_se, concavity_se, concave_points_se,symmetry_se,fractal_dimension_se, radius_worst, texture_worst,
#        perimeter_worst, area_worst, smoothness_worst,
#     compactness_worst, concavity_worst, concave_points_worst,
#        symmetry_worst, fractal_dimension_worst)
# # changing the input_data to numpy array
# input_data_as_numpy_array = np.asarray(input_data) # converting this list into numpy array

# # reshape the numpy array as we are predicting for one instance

# input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
# print(input_data_reshaped)

# predictions = a.predict(input_data_reshaped)
# print(predictions)
# if predictions[0] == 1:
#     st.write("M")
# else:
#     st.write("B")
