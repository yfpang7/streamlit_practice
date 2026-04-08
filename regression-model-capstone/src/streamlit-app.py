import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def read_csv(file_path):
    return pl.read_csv(file_path)

@st.cache_resource(show_spinner='Loading model...')
def load_model(model_path):
    return joblib.load(model_path)

# initialize the session with an empty predicted price
# this is needed to prevent an error when the app first loads and the user hasn't submitted the form yet
if 'predicted_price' not in st.session_state:
    st.session_state['predicted_price'] = None

if __name__ == '__main__':
    st.title('Used Car Price Calculator')

    data = read_csv('regression-model-capstone/data/ca-dealers-used-cleaned.csv')
    ml_model = load_model('model.joblib')

    st.dataframe(data.head(2))

    with st.form('car_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            miles = st.number_input('Miles', min_value=int(data['miles'].min()), max_value=int(data['miles'].max()), value=int(data['miles'].mean()), key='miles')
            model_name = st.selectbox('Model', options=sorted(data['model'].unique().to_list()), key='model')

        with col2:
            year = st.number_input('Year', min_value=int(data['year'].min()), max_value=int(data['year'].max()), value=int(data['year'].mean()), step=1, key='year')
            engine_size = st.number_input('Engine Size (L)', min_value=float(data['engine_size'].min()), max_value=float(data['engine_size'].max()), value=float(data['engine_size'].mean()), step=0.1, key='engine_size')

        with col3:
            make = st.selectbox('Make', options=sorted(data['make'].unique().to_list()), key='make')
            state = st.selectbox('Province/State', options=sorted(data['state'].unique().to_list()), key='state')

        button = st.form_submit_button(label='Calculate Price')

    if button:
        input_df = pd.DataFrame([{
            'miles':       miles,
            'year':        year,
            'make':        make,
            'model':       model_name,
            'engine_size': engine_size,
            'state':       state,
        }])
        st.write('### Input Data')
        st.dataframe(input_df)
        st.write(st.session_state)
        predicted_price = ml_model.predict(input_df)[0]
        st.session_state['predicted_price'] = predicted_price
        st.success(f'### Estimated Price: ${predicted_price:,.0f}')
