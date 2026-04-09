import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def train_model():

    df = pd.read_csv("data/new.csv")

    df = df.dropna()

    # Encode categorical features
    df = pd.get_dummies(df)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=5, random_state=42)

    clusters = model.fit_predict(X_scaled)

    df["Cluster"] = clusters

    return df, model, scaler