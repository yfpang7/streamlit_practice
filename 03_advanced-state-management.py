import streamlit as st
from datetime import datetime, timedelta

st.title('Advanced State Management')

# store widget value in session state
st.subheader('Store widget value in session state')

# create a slider and store its value in session state
st.slider('Select a number', 0, 10, key='slider_value')
st.write(st.session_state.slider_value)

# initialize widget value with session state
st.subheader('Initialize widget value with session state')

if 'num_input' not in st.session_state:
    st.session_state['num_input'] = 5

st.number_input('Enter a number', 0, 10, key='num_input')
st.write('The value of the number input is:', st.session_state.num_input)

# callbacks
# functions that run when a widget value changes
# timerange functionality is a common use case for callbacks, where the start and end date inputs are only shown when the user selects 'custom' range
st.subheader('Use callbacks')

st.markdown('### Select your time range')

def add_timedelta():
    initial = st.session_state.start_date

    if st.session_state.radio_range == '7 Days':
        st.session_state.end_date = initial + timedelta(days=7)
    elif st.session_state.radio_range == '28 days':
        st.session_state.end_date = initial + timedelta(days=28)
    # if the user selects 'custom', we don't want to update the end date, so we can just pass
    else:
        pass

def subtract_timedelta():
    end = st.session_state.end_date

    if st.session_state.radio_range == '7 Days':
        st.session_state.start_date = end - timedelta(days=7)
    elif st.session_state.radio_range == '28 days':
        st.session_state.start_date = end - timedelta(days=28)
    else:
        pass

st.radio('Select a range', ['7 Days', '28 days', 'custom'], horizontal=True, key='radio_range', on_change=add_timedelta)
st.write('The selected range is:', st.session_state.radio_range)


col1, col2, col3 = st.columns(3)


col1.date_input('Start date', key='start_date', on_change=add_timedelta, value=datetime.now())
col2.date_input('End date', key='end_date', on_change=subtract_timedelta)

st.write('The start date is:', st.session_state.start_date)
st.write('The end date is:', st.session_state.end_date)