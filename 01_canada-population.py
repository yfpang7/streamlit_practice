import streamlit as st
import polars as pl
from plotnine import *
import plotly.express as px
import numpy as np

st.title("Population of Canada")
st.write("Source table can be found [here](https://github.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/blob/master/12_dashboard_capstone/data/quarterly_canada_population.csv)")

# import the data
canada_population = pl.read_csv("streamlit-fundamentals/quarterly_canada_population.csv")

# expand the quarter column into year and quarter
canada_population = canada_population.with_columns(
    pl.col('Quarter').str.split(' ').list.get(0).alias('Quarter'),
    pl.col('Quarter').str.split(' ').list.get(1).cast(pl.Int32).alias('Year')
)

# expander
with st.expander("Show raw data"):
    st.dataframe(canada_population)

# form
with st.form('population-form'):
    # columns
    col1, col2, col3 = st.columns(3)

    # column 1
    with col1:
        col1.write('Choose a starting date')
        start_quarter = col1.selectbox(
            "Quarter",
            options=canada_population["Quarter"].unique(),
            index=2, key='start_q'
        )
        start_year = col1.slider(
            'Year', 
            min_value=canada_population['Year'].min(), 
            max_value=canada_population['Year'].max(), 
            value=1991, step=1, key='start_y'
        )

    # column 2
    with col2:
        col2.write('Choose an end date')
        end_quarter = (
            col2
            .selectbox(
                'Quarter',
                options=canada_population["Quarter"].unique(),
                index=0, 
                key='end_q'
            )
        )
        end_year = col2.slider('Year', 
                               min_value=canada_population['Year'].min(), 
                               max_value=canada_population['Year'].max(), 
                               value=2023, step=1, 
                               key='end_y'
        )

    # column 3
    with col3:
        col3.write('Choose a location')
        location = (
            col3
            .selectbox(
                'Choose a location',
                options=(
                    canada_population
                    .drop(['Quarter', 'Year',])
                    .columns
                ),
                index=0
            )
        )

    analyze = st.form_submit_button("Analyze", type='primary')
    
# Q1 of 1991 will be 1991.0; if Q2 then 1991.25
def format_date_for_comparison(quarter, year):
    offsets = {"Q1": 0.0, "Q2": 0.25, "Q3": 0.50, "Q4": 0.75}
    return year + offsets.get(quarter, 0.0)

def end_before_start(start_quarter, start_year, end_quarter, end_year):
    return format_date_for_comparison(start_quarter, start_year) > \
           format_date_for_comparison(end_quarter, end_year)

start_num = format_date_for_comparison(start_quarter, start_year)
end_num = format_date_for_comparison(end_quarter, end_year)

# check if the selection exists
# check the number of rows if it is more than 0 means there are rows otherwise none
start_exists = canada_population.filter(
    (pl.col('Quarter') == start_quarter) & (pl.col('Year') == start_year)
).height > 0
end_exists = canada_population.filter(
    (pl.col('Quarter') == end_quarter) & (pl.col('Year') == end_year)
).height > 0

# check if the start date is after end date then produce this error
if end_before_start(start_quarter, start_year, end_quarter, end_year):
    st.error("End date must be after start date.")
    
elif not start_exists or not end_exists:
    st.error('No data available. Check your quarter and year selection')
    
else:
    start_num = format_date_for_comparison(start_quarter, start_year)
    end_num = format_date_for_comparison(end_quarter, end_year)

    filtered_df = (
        canada_population
        .with_columns(
            (pl.col('Year') + pl.when(pl.col('Quarter') == 'Q1').then(pl.lit(0.0))
            .when(pl.col('Quarter') == 'Q2').then(pl.lit(0.25))
            .when(pl.col('Quarter') == 'Q3').then(pl.lit(0.50))
            .otherwise(pl.lit(0.75))).alias('sort_key')
        )
        .filter((pl.col('sort_key') >= start_num) & (pl.col('sort_key') <= end_num))
        .sort('sort_key')
    )
    filtered_df = filtered_df.with_columns(
        (pl.col('Quarter') + ' ' + pl.col('Year').cast(pl.String)).alias('label')
    )

    # tabs
    tab1, tab2 = st.tabs(['Population change', 'Compare'])

    with tab1:
        st.subheader(f'Population change from {start_quarter} {start_year} to {end_quarter} {end_year}')

        col1, col2 = st.columns(2)

        with col1:
            initial = (
                canada_population
                .filter((pl.col('Quarter') == start_quarter) & (pl.col('Year') == start_year))
                [location]
                .item()
            )
            final = (
                canada_population
                .filter((pl.col('Quarter') == end_quarter) & (pl.col('Year') == end_year))
                [location]
                .item()
            )
            percentage_diff = np.round((final - initial) / initial * 100, 2)
            delta = f"{percentage_diff}%"

            st.metric(label=f'{start_quarter} {start_year}', value=initial)
            st.metric(label=f'{end_quarter} {end_year}', value=final, delta=delta)

        with col2:
            fig = px.line(
                filtered_df.to_pandas(),
                x='label', y=location,
                labels={'label': 'Time', location: 'Population'}
            )
            fig.update_xaxes(tickvals=[
                filtered_df['label'][0],
                filtered_df['label'][-1]
            ])
            st.plotly_chart(fig)

    with tab2:
        st.subheader('Compare with other locations')

        location_cols = canada_population.drop(['Quarter', 'Year']).columns
        all_targets = st.multiselect(
            "Choose other locations",
            options=location_cols,
            default=[location]
        )

        if all_targets:
            fig = px.line(
                filtered_df.to_pandas(),
                x='label', y=all_targets,
                labels={'label': 'Time', 'value': 'Population', 'variable': 'Location'}
            )
            fig.update_xaxes(tickvals=[
                filtered_df['label'][0],
                filtered_df['label'][-1]
            ])
            st.plotly_chart(fig)