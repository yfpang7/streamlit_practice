import streamlit as st
import polars as pl
from plotnine import *
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

# read the config file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# render the authentication form
authenticator.login('sidebar')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data()
def load_data(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


segmentation_data = load_data('../data/segmentation data.csv')

# numpy feature matrix used by sklearn (excludes ID and Income)
X = segmentation_data.select(pl.exclude('ID', 'Income')).to_numpy()


# ---------------------------------------------------------------------------
# Data Science – experiment & plots
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner='Calculating silhouette scores...')
def run_experiment(_X):
    k_silhouettes, agglo_silhouettes, inertias = [], [], []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        k_labels = kmeans.fit_predict(_X)
        k_silhouettes.append(silhouette_score(_X, k_labels))
        inertias.append(float(kmeans.inertia_))

        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        a_labels = agglo.fit_predict(_X)
        agglo_silhouettes.append(silhouette_score(_X, a_labels))

    return pl.DataFrame({
        'n_clusters': list(range(2, 11)),
        'kmeans': k_silhouettes,
        'agglo': agglo_silhouettes,
        'inertia': inertias,
    })


def plot_silhouette(metrics_df: pl.DataFrame):
    melted = (
        metrics_df
        .select(['n_clusters', 'kmeans', 'agglo'])
        .unpivot(index='n_clusters', on=['kmeans', 'agglo'],
                 variable_name='algorithm', value_name='silhouette')
    )
    return (
        ggplot(melted.to_pandas(), aes(x='n_clusters', y='silhouette', color='algorithm'))
        + geom_line(size=1)
        + geom_point(size=3)
        + scale_x_continuous(breaks=list(range(2, 11)))
        + labs(title='Silhouette Score by Number of Clusters',
               x='Number of Clusters', y='Silhouette Score', color='Algorithm')
        + theme_minimal()
    )


def plot_elbow(metrics_df: pl.DataFrame):
    return (
        ggplot(metrics_df.to_pandas(), aes(x='n_clusters', y='inertia'))
        + geom_line(size=1, color='steelblue')
        + geom_point(size=3, color='steelblue')
        + scale_x_continuous(breaks=list(range(2, 11)))
        + labs(title='Elbow Curve (KMeans Inertia)',
               x='Number of Clusters', y='Inertia')
        + theme_minimal()
    )


def display_ds_content():
    st.subheader('Dataset Preview')
    st.dataframe(segmentation_data)

    if st.button('Run Experiment'):
        metrics_df = run_experiment(X)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Silhouette Scores')
            st.dataframe(metrics_df.select(['n_clusters', 'kmeans', 'agglo']))
            st.pyplot(plot_silhouette(metrics_df).draw())
        with col2:
            st.subheader('Elbow Curve')
            st.dataframe(metrics_df.select(['n_clusters', 'inertia']))
            st.pyplot(plot_elbow(metrics_df).draw())

        best_k = metrics_df.sort('kmeans', descending=True).row(0)[0]
        st.info(f'Best number of clusters by KMeans silhouette score: **{best_k}**')


# ---------------------------------------------------------------------------
# Marketing – clustering & plots
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner='Running KMeans...')
def get_cluster_df(num_clusters: int) -> pl.DataFrame:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X).tolist()
    return segmentation_data.with_columns(pl.Series('Cluster', labels))


def plot_income_by_cluster(cluster_df: pl.DataFrame):
    return (
        ggplot(cluster_df.to_pandas(),
               aes(x='factor(Cluster)', y='Income', fill='factor(Cluster)'))
        + geom_boxplot(alpha=0.7)
        + labs(title='Income Distribution by Cluster',
               x='Cluster', y='Income ($)', fill='Cluster')
        + theme_minimal()
        + theme(legend_position='none')
    )


def plot_age_by_cluster(cluster_df: pl.DataFrame):
    return (
        ggplot(cluster_df.to_pandas(),
               aes(x='Age', fill='factor(Cluster)'))
        + geom_histogram(bins=20, alpha=0.7, color='white')
        + facet_wrap('~Cluster', labeller='label_both')
        + labs(title='Age Distribution by Cluster', x='Age', y='Count', fill='Cluster')
        + theme_minimal()
        + theme(legend_position='none')
    )


def plot_sex_by_cluster(cluster_df: pl.DataFrame):
    sex_counts = (
        cluster_df
        .with_columns(
            pl.when(pl.col('Sex') == 0)
            .then(pl.lit('Male'))
            .otherwise(pl.lit('Female'))
            .alias('Gender')
        )
        .group_by(['Cluster', 'Gender'])
        .agg(pl.len().alias('count'))
    )
    return (
        ggplot(sex_counts.to_pandas(),
               aes(x='factor(Cluster)', y='count', fill='Gender'))
        + geom_col(position='fill')
        + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}%' for v in l])
        + labs(title='Gender Breakdown by Cluster',
               x='Cluster', y='Proportion', fill='Gender')
        + theme_minimal()
    )


def plot_education_by_cluster(cluster_df: pl.DataFrame):
    edu_counts = (
        cluster_df
        .with_columns(
            pl.when(pl.col('Education') == 1)
            .then(pl.lit('High School'))
            .when(pl.col('Education') == 2)
            .then(pl.lit("Bachelor's"))
            .otherwise(pl.lit("Postgraduate"))
            .alias('Education_label')
        )
        .group_by(['Cluster', 'Education_label'])
        .agg(pl.len().alias('count'))
    )
    return (
        ggplot(edu_counts.to_pandas(),
               aes(x='factor(Cluster)', y='count', fill='Education_label'))
        + geom_col(position='fill')
        + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}%' for v in l])
        + labs(title='Education Breakdown by Cluster',
               x='Cluster', y='Proportion', fill='Education')
        + theme_minimal()
    )


def plot_occupation_by_cluster(cluster_df: pl.DataFrame):
    occ_counts = (
        cluster_df
        .with_columns(
            pl.when(pl.col('Occupation') == 0)
            .then(pl.lit('Unemployed'))
            .when(pl.col('Occupation') == 1)
            .then(pl.lit('Skilled'))
            .otherwise(pl.lit('Highly Skilled'))
            .alias('Occupation_label')
        )
        .group_by(['Cluster', 'Occupation_label'])
        .agg(pl.len().alias('count'))
    )
    return (
        ggplot(occ_counts.to_pandas(),
               aes(x='factor(Cluster)', y='count', fill='Occupation_label'))
        + geom_col(position='dodge')
        + labs(title='Occupation by Cluster',
               x='Cluster', y='Count', fill='Occupation')
        + theme_minimal()
    )


def display_group_metrics(cluster_df: pl.DataFrame, num_clusters: int):
    for i in range(num_clusters):
        g = cluster_df.filter(pl.col('Cluster') == i)

        male_pct = round(g.filter(pl.col('Sex') == 0).height / g.height, 2) * 100
        female_pct = 100 - male_pct
        married_pct = round(g.filter(pl.col('Marital status') == 1).height / g.height, 2) * 100

        mean_age = round(g['Age'].mean(), 1)
        min_age = g['Age'].min()
        max_age = g['Age'].max()

        hs_pct = round(g.filter(pl.col('Education') == 1).height / g.height, 2) * 100
        uni_pct = round(g.filter(pl.col('Education').is_in([2, 3])).height / g.height, 2) * 100

        mean_income = round(g['Income'].mean(), 0)
        min_income = g['Income'].min()
        max_income = g['Income'].max()

        occ_mode = g['Occupation'].mode()[0]
        employment = {0: 'Unemployed', 1: 'Skilled employee', 2: 'Highly skilled employee'}.get(occ_mode, 'Unknown')

        city_mode = g['Settlement size'].mode()[0]
        city = {0: 'Small city', 1: 'Mid-sized city', 2: 'Large city'}.get(city_mode, 'Unknown')

        with st.container(border=True):
            st.subheader(f'Cluster {i + 1}  —  {g.height} customers')
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('**Demographics**')
                st.metric('Men', f'{male_pct}%')
                st.metric('Women', f'{female_pct}%')
                st.metric('Married', f'{married_pct}%')
            with c2:
                st.markdown('**Age & Income**')
                st.metric('Mean Age', mean_age)
                st.metric('Age Range', f'{min_age}–{max_age}')
                st.metric('Mean Income', f'${mean_income:,.0f}')
                st.metric('Income Range', f'${min_income:,}–${max_income:,}')
            with c3:
                st.markdown('**Profile**')
                st.metric('High School', f'{hs_pct}%')
                st.metric('University', f'{uni_pct}%')
                st.write(f'**Employment:** {employment}')
                st.write(f'**City size:** {city}')


def display_marketing_content():
    st.subheader('Dataset Preview')
    st.dataframe(segmentation_data)

    num_clusters = st.slider('Number of clusters', 2, 10, value=4)

    if st.button('Generate Clusters'):
        cluster_df = get_cluster_df(num_clusters)

        st.subheader('Cluster Visualisations')
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_income_by_cluster(cluster_df).draw())
        with col2:
            st.pyplot(plot_age_by_cluster(cluster_df).draw())

        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(plot_sex_by_cluster(cluster_df).draw())
        with col4:
            st.pyplot(plot_education_by_cluster(cluster_df).draw())

        st.pyplot(plot_occupation_by_cluster(cluster_df).draw())

        st.subheader('Cluster Profiles')
        display_group_metrics(cluster_df, num_clusters)


# ---------------------------------------------------------------------------
# Auth routing
# ---------------------------------------------------------------------------

if st.session_state['authentication_status']:
    authenticator.logout('Logout', 'sidebar', key='unique_key')
    st.write(f'Welcome *{st.session_state["name"]}*')

    if st.session_state['username'] == 'marketing':
        st.title('Marketing Campaign Dashboard')
        display_marketing_content()

    elif st.session_state['username'] == 'datascience':
        st.title('Data Science Dashboard')
        display_ds_content()

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')