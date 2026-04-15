import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# load the config file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)