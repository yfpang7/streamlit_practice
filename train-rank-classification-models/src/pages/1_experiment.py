import streamlit as st
import polars as pl
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# --- Constants ---
MODEL_OPTIONS = ['Baseline', 'Decision Tree', 'Random Forest', 'Gradient Boosted Classifier']
MAX_FEATURES = 13

# --- Page Config ---
st.set_page_config(
    page_title='Experiment',
    page_icon='🧪',
    layout='centered',
)


# --- Data & Model Functions ---
@st.cache_data
def load_data() -> pl.DataFrame:
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return pl.from_pandas(df)


@st.cache_data
def split_data(df: pl.DataFrame):
    X = df.drop('target').to_numpy()
    y = df['target'].to_numpy()
    return train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)


@st.cache_data
def select_features(X_train, y_train, X_test, k: int):
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel


@st.cache_data(show_spinner='Training and evaluating model...')
def fit_and_score(model_name: str, k: int) -> float:
    model_map = {
        'Baseline': DummyClassifier(strategy='stratified', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosted Classifier': GradientBoostingClassifier(random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f'Unknown model: {model_name}')

    X_train_sel, X_test_sel = select_features(X_train, y_train, X_test, k)
    clf = model_map[model_name]
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    return f1_score(y_test, y_pred, average='weighted')


# --- Session State Initialisation ---
if 'dataset' not in st.session_state:
    st.session_state.dataset = load_data()

for key in ('score', 'model', 'num_features'):
    if key not in st.session_state:
        st.session_state[key] = []

# --- Prepare Data ---
df = st.session_state.dataset
X_train, X_test, y_train, y_test = split_data(df)


# --- Callback ---
def save_performance(model_name: str, num_features: int) -> None:
    score = fit_and_score(model_name, num_features)
    st.session_state.score.append(score)
    st.session_state.model.append(model_name)
    st.session_state.num_features.append(num_features)


# --- UI ---
st.title('🧪 Experiment')

col1, col2 = st.columns(2)

with col1:
    st.selectbox('Choose a model', MODEL_OPTIONS, key='model_select')

with col2:
    st.number_input(
        'Number of features to keep',
        min_value=1,
        max_value=MAX_FEATURES,
        step=1,
        key='num_features_select',
    )

st.button(
    'Train Model',
    type='primary',
    on_click=save_performance,
    args=(st.session_state.model_select, st.session_state.num_features_select),
)

if st.session_state.score:
    st.success(f'Last trained F1-score: {st.session_state.score[-1]:.4f}')

with st.expander('Inspect Dataset'):
    st.dataframe(df)