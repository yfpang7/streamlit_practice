import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from plotnine import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

# URL = "https://raw.githubusercontent.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/master/17_caching_capstone/data/mushrooms.csv"
URL = 'https://raw.githubusercontent.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/master/18_caching_capstone/data/mushrooms.csv'

COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']

FEATURE_COLS = ['odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'ring-type', 'spore-print-color']

# Function to read the data
@st.cache_data
def read_data(input):
    return pl.read_csv(input)

# create a function to select the X
def drop_label(data) -> pl.Series:
    return data.drop('class').columns

# create a preprocessor
def preprocessor(ordinal_cols):
    preprocessor = ColumnTransformer(
        transformers=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
        ('label', OrdinalEncoder(dtype=int), ['class'])
        ],
        remainder='passthrough'
    )
    return preprocessor

# create the function for pipeline
def pipeline(ordinal_cols):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor(ordinal_cols=ordinal_cols))
        ]
    )
    pipeline.set_output(transform='pandas')

    return pipeline

# Transform the data for training
@st.cache_data
def transforming_pipeline(data):
    # data prep
    ordinal_cols = drop_label(data)
    pipe = pipeline(ordinal_cols)
    # transform the data
    transformed_dataframe_pd = pipe.fit_transform(data.to_pandas())
    transformed_dataframe_pl = pl.from_pandas(transformed_dataframe_pd)
    # renaming
    transformed_dataframe = transformed_dataframe_pl.rename({c: c.split("__")[-1] for c in transformed_dataframe_pl.columns})

    # Fit a dedicated encoder on only FEATURE_COLS for encoding prediction input
    pred_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    pred_encoder.fit(data.to_pandas()[FEATURE_COLS])

    return transformed_dataframe, pred_encoder

# train the model
@st.cache_data(show_spinner="Training the model... This may take a while.")
def train_the_model(transformed_dataframe):
    '''
    Only 9 features will be used for the machine learning:
    
    '''
    X = transformed_dataframe.select(FEATURE_COLS).to_pandas()
    y = transformed_dataframe['class'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=y, shuffle=True,random_state=42)
    
    # create model 
    gbc = GradientBoostingClassifier(random_state=42)
    model_pipeline = Pipeline(
        [
            ('gbc', gbc),
        ]
    )
    # hyperparameter grid
    param_grid = {
        "gbc__n_estimators":     [100, 200, 300],
        "gbc__max_depth":        [3, 5, 7],
        "gbc__learning_rate":    [0.05, 0.1, 0.2],
        "gbc__subsample":        [0.8, 1.0],
        "gbc__min_samples_leaf": [1, 5, 10],
    }
    # cv strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    )
    # fit the model
    return search.fit(X_train, y_train)

# Function to make a prediction
def make_prediction(selections, model):
    best_model = model.best_estimator_
    predictions = best_model.predict(selections)
    return predictions

if __name__ == "__main__":
    st.title("Mushroom classifier 🍄")
    
    # Read the data
    data = read_data(URL)

    if data is not None:
        with st.expander('Inspect Mushroom Dataframe'):
            st.dataframe(data)
    
    st.subheader("Step 1: Select the values for prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        odor = st.selectbox('Odor', ('a - almond', 'l - anisel', 'c - creosote', 'y - fishy', 'f - foul', 'm - musty', 'n - none', 'p - pungent', 's - spicy'))
        stalk_surface_above_ring = st.selectbox('Stalk surface above ring', ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        stalk_color_below_ring = st.selectbox('Stalk color below ring', ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange', 'p - pink', 'e - red', 'w - white', 'y - yellow'))
    with col2:
        gill_size = st.selectbox('Gill size', ('b - broad', 'n - narrow'))
        stalk_surface_below_ring = st.selectbox('Stalk surface below ring', ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        ring_type = st.selectbox('Ring type', ('e - evanescente', 'f - flaring', 'l - large', 'n - none', 'p - pendant', 's - sheathing', 'z - zone'))
    with col3:
        gill_color = st.selectbox('Gill color', ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'g - gray', 'r - green', 'o - orange', 'p - pink', 'u - purple', 'e - red', 'w - white', 'y - yellow'))
        stalk_color_above_ring = st.selectbox('Stalk color above ring', ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange', 'p - pink', 'e - red', 'w - white', 'y - yellow'))
        spore_print_color = st.selectbox('Spore print color', ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'r - green', 'o - orange', 'u - purple', 'w - white', 'y - yellow'))

    st.subheader("Step 2: Ask the model for a prediction")

    pred_btn = st.button("Predict", type="primary")

    # If the button is clicked:
    if pred_btn:
        temp_data, pred_encoder = transforming_pipeline(data)
        # st.dataframe(temp_data.head())
        gridcsv = train_the_model(temp_data)

        # st.write(f"Test F1 (weighted): {gridcsv.best_estimator_}")

        # Build the prediction input: extract the letter code (first char) from each selectbox value
        x_pred = pd.DataFrame([{
            'odor':                      odor[0],
            'gill-size':                 gill_size[0],
            'gill-color':                gill_color[0],
            'stalk-surface-above-ring':  stalk_surface_above_ring[0],
            'stalk-surface-below-ring':  stalk_surface_below_ring[0],
            'stalk-color-above-ring':    stalk_color_above_ring[0],
            'stalk-color-below-ring':    stalk_color_below_ring[0],
            'ring-type':                 ring_type[0],
            'spore-print-color':         spore_print_color[0],
        }])

        # Encode using the dedicated encoder fitted only on FEATURE_COLS
        # it converts the raw letters into the same integers the model expects 
        # making the prediction input compatible with the trained model.
        x_pred_encoded = pd.DataFrame(
            pred_encoder.transform(x_pred[FEATURE_COLS]),
            columns=FEATURE_COLS
        )

        # 5. Make a prediction
        pred = make_prediction(x_pred_encoded, gridcsv)

        # 6. Format the prediction to be a nice text
        if pred is not None:
            prediction_text = "Edible" if pred[0] == 0 else "Poisonous"
            # 7. Output it to the screen
            st.write(f"The mushroom is predicted to be: {prediction_text}")




