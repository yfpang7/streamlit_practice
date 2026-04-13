import streamlit as st
import polars as pl

st.set_page_config(
    page_title='Homepage',
    page_icon='🏘️',
    layout='centered',
)

# --- Session State Initialisation ---
for key in ('score', 'model', 'num_features'):
    if key not in st.session_state:
        st.session_state[key] = []


def display_ranked_df() -> None:
    ranked_df = pl.DataFrame({
        'model': st.session_state.model,
        'num_features': st.session_state.num_features,
        'F1-score': st.session_state.score,
    })
    st.dataframe(ranked_df.sort('F1-score', descending=True))


st.title('🏆 Model Ranking')
st.subheader('Train a model in the next page to see the results 👉')

if st.session_state.model:
    display_ranked_df()
else:
    st.info('No models trained yet. Please train a model on the next page.')