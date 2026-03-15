import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


st.title("Customer Segmentation Dashboard")

# Load dataset
df = pd.read_csv("new.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())


# ==========================
# DATA PREPROCESSING
# ==========================

# Select numeric columns
numeric_df = df.select_dtypes(include=['int64','float64'])

# Handle missing values
numeric_df = numeric_df.fillna(numeric_df.mean())

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(numeric_df)


# ==========================
# TRAIN MODEL
# ==========================

kmeans = KMeans(n_clusters=5, random_state=42)

df["Cluster"] = kmeans.fit_predict(X)


# ==========================
# CLUSTER DISTRIBUTION
# ==========================

st.subheader("Customer Cluster Distribution")

cluster_counts = df["Cluster"].value_counts()

st.bar_chart(cluster_counts)


# ==========================
# CLUSTER VISUALIZATION
# ==========================

st.subheader("Customer Segmentation Visualization")

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X)

pca_df = pd.DataFrame()

pca_df["x"] = pca_data[:,0]
pca_df["y"] = pca_data[:,1]
pca_df["cluster"] = df["Cluster"]

fig = plt.figure()

sns.scatterplot(
    x="x",
    y="y",
    hue="cluster",
    data=pca_df,
    palette="Set2"
)

st.pyplot(fig)


# ==========================
# CUSTOMER SEGMENT TABLE
# ==========================

st.subheader("Customer Segments Table")

st.dataframe(df)


# ==========================
# MARKETING RECOMMENDATION
# ==========================

st.subheader("Marketing Strategy Recommendation")

cluster_choice = st.selectbox(
    "Select Customer Cluster",
    sorted(df["Cluster"].unique())
)

def recommendation(cluster):

    strategies = {
        0: "VIP Customers → Offer Premium Membership",
        1: "Loyal Customers → Loyalty Rewards",
        2: "Potential Customers → Personalized Promotions",
        3: "Low Engagement → Discount Campaign",
        4: "At Risk Customers → Retention Marketing"
    }

    return strategies.get(cluster, "General Marketing Strategy")

st.success(recommendation(cluster_choice))


# ==========================
# CUSTOMER PREDICTION
# ==========================

st.subheader("Predict New Customer Segment")

input_data = []

for col in numeric_df.columns[:5]:
    value = st.number_input(col)
    input_data.append(value)

if st.button("Predict Customer Segment"):

    sample = [input_data]

    sample_scaled = scaler.transform(sample)

    prediction = kmeans.predict(sample_scaled)

    st.success(f"Customer belongs to Segment {prediction[0]}")