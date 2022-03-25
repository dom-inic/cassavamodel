import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from cassava.pretrained import get_model
import numpy as np
from PIL import Image
import pandas as pd
import cv2 
from st_aggrid import AgGrid
import plotly.express as px
import io 


with st.sidebar:
    choose = option_menu("Main Menu", ["Home","About", "Data_Exploration", "Contact"],
                         icons=['house','house', 'bar-chart-line','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "black"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "black"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

logo = Image.open(r'pics/healthy.png')
profile = Image.open(r'pics/healthy.png')

if choose == "Home":
    st.set_option("deprecation.showfileUploaderEncoding", False)


    @st.cache(allow_output_mutation=True)
    def load_model(name):
        model = get_model(name=name)
        return model


    model = load_model("tf_efficientnet_b4")

    st.write(
        """
        # Cassava Leaf Disease Classification
        """
    )

    file = st.file_uploader("Upload your Image Here", type=["jpg", "png"])

    def make_prediction(image, model):
        img = np.array(image)
        value = model.predit_as_json(img)
        return {
            "class_name": value['class_name'],
            "confidence": str(value['confidence'])

        }

    if file is None:
        st.text("Please Upload an image file")
    else:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)
        prediction = make_prediction(image=image, model=model)
        st.json(prediction)
        st.success("Prediction made sucessful")
elif choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About the Project</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    
    st.write("CLD Model is a model build using pytorch, opencv, sklearn, pandas, numpy and many more other machine learning algorithms. The Dataset used to train this classifier model can be found through this link: https://www.kaggle.com/c/cassava-leaf-disease-classification/data. This User Interface is build using streamlit, Streamlit enables data scientists and machine learning practitioners to build data and machine learning applications quickly")    
    # st.image(profile, width=130)


elif choose == "Data_Exploration":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Data Exploration</p>', unsafe_allow_html=True)

    st.subheader('Import Data into Python')
    st.markdown('To start a data science project in Python, you will need to first import your data into a Pandas data frame. Often times we have our raw data stored in a local folder in csv format. Therefore let\'s learn how to use Pandas\' read_csv method to read our sample data into Python.')

    #Display the first code snippet
    code = '''import pandas as pd #import the pandas library\ndf=pd.read_csv(r'data/train.csv') #read the csv file into pandas\ndf.head() #display the first 5 rows of the data'''
    st.code(code, language='python')

    #Allow users to check the results of the first code snippet by clicking the 'Check Results' button
    df=pd.read_csv(r'data/train.csv')
    df_head=df.head()
    if st.button('Check Results', key='1'):
        st.write(df_head)
    else:
        st.write('---')

    #Display the second code snippet
    code = '''df.tail() #display the last 5 rows of the data'''
    st.code(code, language='python')

    #Allow users to check the results of the second code snippet by clicking the 'Check Results' button
    df=pd.read_csv(r'data/train.csv')
    df_tail=df.tail()
    if st.button('Check Results', key='2'):
        st.write(df_tail)
    else:
        st.write('---')     

    #Display the third code snippet
    st.write('   ')
    st.markdown('After we import the data into Python, we can use the following code to check the information about the data frame, such as number of rows and columns, data types for each column, etc.')
    code = '''df.info()''' 
    st.code(code, language='python')

    #Allow users to check the results of the third code snippet by clicking the 'Check Results' button
    import io 
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    if st.button('Check Results', key='3'):
        st.text(s)
    else:
        st.write('---')

elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')