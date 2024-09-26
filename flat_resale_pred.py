import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate,KFold
from sklearn.model_selection import cross_val_score,KFold
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from streamlit_option_menu import option_menu


with open('Regression_model.pkl',"rb") as file:
    regg_model = pickle.load(file)

month_list = ['January','February','March','April','May','June','July','August','Septemper','October','November','December']

month_list_encoded = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'Septemper':9,'October':10,'November':11,
                      'December':12}

year_list = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
             2012, 2015, 2016, 2013, 2014, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

town_list = ['ANG MO KIO','BEDOK','BISHAN','BUKIT BATOK','BUKIT MERAH','BUKIT TIMAH','CENTRAL AREA','CHOA CHU KANG','CLEMENTI','GEYLANG','HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH',
              'WOODLANDS','YISHUN','LIM CHU KANG','SEMBAWANG','BUKIT PANJANG','PASIR RIS','PUNGGOL']

town_list_encoded = {'ANG MO KIO' : 0 ,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,'BUKIT MERAH' : 4,'BUKIT PANJANG' : 5,'BUKIT TIMAH' : 6,
        'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
        'KALLANG/WHAMPOA' : 14,'LIM CHU KANG' : 15,'MARINE PARADE' : 16,'PASIR RIS' : 17,'PUNGGOL' : 18,'QUEENSTOWN' : 19,
        'SEMBAWANG' : 20,'SENGKANG' : 21,'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26}


flat_type_list = ['1 ROOM','3 ROOM', '4 ROOM','5 ROOM' ,'2 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']

flat_type_encoded={'1 ROOM': 0,'2 ROOM' : 1,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,'EXECUTIVE' : 5,'MULTI-GENERATION' : 6}

flat_model_list = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED', 'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT', 'ADJOINED FLAT', 'PREMIUM MAISONETTE',
                    'MODEL A2', 'TYPE S1', 'TYPE S2', 'DBSS', 'PREMIUM APARTMENT LOFT', '3GEN'] 

flat_model_list_encoded = {'2-ROOM' : 0,'3GEN' : 1,'ADJOINED FLAT' : 2,'APARTMENT' : 3,'DBSS' : 4,'IMPROVED' : 5,'IMPROVED-MAISONETTE' : 6,
                'MAISONETTE' : 7,'MODEL A' : 8,'MODEL A-MAISONETTE' : 9,'MODEL A2': 10,'MULTI GENERATION' : 11,'NEW GENERATION' : 12,
                'PREMIUM APARTMENT' : 13,'PREMIUM APARTMENT LOFT' : 14,'PREMIUM MAISONETTE' : 15,'SIMPLIFIED' : 16,'STANDARD' : 17,
                'TERRACE' : 18,'TYPE S1' : 19,'TYPE S2' : 20}




def re_sale_price(input_data):
    input_data_array = np.array(input_data)
    re_sale_price_prediction = regg_model.predict(input_data)
    return re_sale_price_prediction
     
st.set_page_config(layout='wide')

title_text = '''<h1 style='font-size : 55px;text-align:center;color:purple;background-color:lightgrey;'>Flat Re-Sale Price Prediction</h1>'''
st.markdown(title_text,unsafe_allow_html=True)

with st.sidebar:

    select = option_menu('MAIN MENU',['HOME','ABOUT','PREDICTION'])

if select == 'HOME':

    st.write(" ")
    st.write(" ")
    st.header(":violet[Housing and Development Board - An Overview]")

    with st.container(border=True):
        st.markdown('''<h5 style='color:#00ffff;font-size:21px'> The Housing & Development Board (HDB; often referred to as the Housing Board),
                    is a statutory board under the Ministry of National Development responsible for the public housing in Singapore.
                     Established in 1960 as a result of efforts in the late 1950s to set up an authority to take over 
                    the Singapore Improvement Trust's (SIT) public housing responsibilities, the HDB focused on the construction 
                    of emergency housing and the resettlement of kampong residents into public housing in the first few years of its existence.''',
                    unsafe_allow_html=True)
        
        st.write(" ")

    st.header(":red[Vision]")

    with st.container(border=True):
        st.markdown('''<h5 style='color:#ff1a66;font-size:28px'>An outstanding organisation creating endearing homes all are proud of.''',unsafe_allow_html=True)

        st.write(" ")

    st.header(":blue[Mission]")

    with st.container(border=True):
        st.markdown('''<h5 style='color:#b3b300;font-size:28px'>We provide affordable, quality housing and a great living environment where communities thrive.''',unsafe_allow_html=True)

        st.write(" ")

    st.header(":red[Services]")

    with st.container(border=True):
        st.markdown('''<h5 style='color:#ff1a66;font-size:28px'>Residential <br> Buisness <br> Car parks <br> General''',unsafe_allow_html=True)

    st.link_button(":violet[** HDB Link**]",url='https://www.hdb.gov.sg/cs/infoweb/homepage',use_container_width = True)

elif select == 'ABOUT':

    st.write(" ")
    st.write(" ")

    st.markdown('''<h6 style ='color:#ff1a66;font-size:31px'><br>Project Title : Singapore Resale Flat Prices Predicting''',unsafe_allow_html=True)


    st.markdown('''<h6 style ='color:#007acc;font-size:31px'><br>Domain : RealEstate ''',unsafe_allow_html=True)

    st.markdown('''<h6 style ='color:#00ffff;font-size:31px'><br>
                Take away Skills : <br>Python Scipting<br>Data Wrangling<br>EDA<br>Model Building<br>Model Deployment in Streamlit and on Render platform''',unsafe_allow_html=True)

elif select == 'PREDICTION':

    title_text = '''<h1 style='font-size: 30px;text-align: center;color:#00ff80;'>To predict the Flat Re-Sale Price, please provide the following information</h1>'''
    st.markdown(title_text, unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    
    
    with col1:

        month = st.selectbox('Month',month_list,index=None)

        town = st.selectbox('Town',town_list,index=None)

        flat_type = st.selectbox('Flat_type',flat_type_list,index=None)
        
        block = st.number_input('Block')

        floor_area_sqm = st.number_input('Floor_Area_sqm')

        flat_model = st.selectbox('Flat_Model',flat_model_list,index=None)

        lease_commence_date = st.number_input('Lease_Start_Date')

               

    with col2:

        remaining_lease = st.number_input('Remaining_Lease')

        year = st.selectbox ('Year',year_list,index=None)  

        age_of_property = st.number_input('Age_of_Property')

        storey = st.number_input ('No of Floors')        
        
        holding_period = st.number_input('Holding_period')

        rate_per_sqm = st.number_input('Rate_per_sqm')

    with col1:

        st.write(" ")
        st.write(" ")

    if st.button(':violet[Predict]',use_container_width=True):
            
            month = month_list_encoded[month]

            town = town_list_encoded[town]

            flat_type = flat_type_encoded[flat_type]

            flat_model = flat_model_list_encoded[flat_model]

            

            prediction = re_sale_price([[month,town,flat_type,block,floor_area_sqm,flat_model,lease_commence_date,remaining_lease,
                                         year,age_of_property,storey,holding_period,rate_per_sqm]])

            st.subheader((f":green[Predicted Resale Price :] {prediction[0]:.2f}"))


    
