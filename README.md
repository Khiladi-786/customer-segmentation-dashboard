# 📊 Customer Segmentation Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> An interactive ML-powered dashboard that segments customers using K-Means clustering and visualizes behavioral patterns through Streamlit — helping businesses design personalized marketing strategies.

**🔗 [Live Demo](https://your-streamlit-link.streamlit.app)** | **📊 [View Dashboard](https://your-streamlit-link.streamlit.app)**

---

## 📌 Project Overview

Customer segmentation is critical for marketing analytics — businesses use it to understand customer diversity and craft targeted campaigns. This project applies **unsupervised machine learning** (K-Means clustering) to group customers based on behavioral and demographic characteristics.

**Key Innovation:** Interactive Streamlit dashboard allowing real-time exploration of customer segments, cluster distributions, and actionable business insights.

---

## 🎯 Key Features

- ✅ **K-Means Clustering** for customer segmentation
- ✅ **Interactive Streamlit Dashboard** with real-time visualizations
- ✅ **PCA-based cluster visualization** in 2D/3D space
- ✅ **Automatic data preprocessing** — handles missing values & outliers
- ✅ **Feature engineering & scaling** using StandardScaler
- ✅ **Cluster insights** — identifies high-value customer segments
- ✅ **Cloud deployment** via Streamlit Cloud
- ✅ **Dataset preview & exploration** tools

---

## 🖼️ Dashboard Preview

The dashboard provides:

| Feature | Description |
|---|---|
| **Dataset Explorer** | Preview raw customer data with statistics |
| **Cluster Visualization** | 2D/3D scatter plots showing customer segments |
| **Distribution Analysis** | Cluster size, centroid analysis, segment characteristics |
| **Actionable Insights** | Business recommendations per segment |

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- Handled missing values using mean/median imputation
- Removed outliers using IQR method
- Encoded categorical variables (one-hot encoding)

### 2. Feature Engineering
- Selected relevant features: age, income, spending score, purchase frequency
- Normalized features using **StandardScaler** for clustering

### 3. K-Means Clustering
- Applied K-Means algorithm with optimal clusters (determined via Elbow Method)
- Grouped customers into distinct behavioral segments

### 4. Dimensionality Reduction
- Used **PCA (Principal Component Analysis)** to visualize high-dimensional clusters in 2D

### 5. Visualization
- Interactive plots using Matplotlib, Seaborn, and Streamlit

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | K-Means clustering, PCA, preprocessing |
| Matplotlib & Seaborn | Static visualizations |
| Streamlit | Interactive web dashboard |
| Streamlit Cloud | Deployment platform |

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Khiladi-786/customer-segmentation-dashboard.git
cd customer-segmentation-dashboard
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
customer-segmentation-dashboard/
│
├── app.py                  # Streamlit dashboard application
├── new.csv                 # Customer dataset
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── model/
│   ├── preprocessing.py    # Data cleaning scripts
│   ├── clustering.py       # K-Means model
│   └── visualization.py    # Plot generation
│
└── outputs/
    └── clusters.csv        # Segmented customer data
```

---

## 🏆 Clustering Results

**Sample Insights from Customer Segments:**

| Segment | Characteristics | Marketing Strategy |
|---|---|---|
| **High-Value Customers** | High income, frequent purchases | Loyalty programs, VIP offers |
| **Budget Shoppers** | Low income, price-sensitive | Discounts, bundle deals |
| **Young Professionals** | Mid income, trending products | Social media campaigns |
| **Inactive Customers** | Low engagement | Re-engagement emails |

---

## 💡 Business Use Cases

Customer segmentation enables businesses to:

- 🎯 **Targeted Marketing:** Personalized campaigns per segment
- 💰 **Revenue Optimization:** Focus on high-value customers
- 📧 **Email Personalization:** Segment-specific messaging
- 🛍️ **Product Recommendations:** Tailor suggestions by group
- 📈 **Customer Retention:** Identify at-risk segments

---

## 🌐 Deployment

**Live Application:** [Customer Segmentation Dashboard](https://your-streamlit-link.streamlit.app)

Deployed using **Streamlit Cloud** for instant access without local setup.

### Deploy Your Own:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

---

## 🔮 Future Enhancements

- [ ] **Elbow Method Integration** — automatic optimal cluster selection
- [ ] **DBSCAN/Hierarchical Clustering** — compare multiple algorithms
- [ ] **Recommendation Engine** — suggest marketing strategies per segment
- [ ] **Real-time Data Integration** — connect to live databases
- [ ] **Customer Lifetime Value (CLV)** prediction module
- [ ] **Advanced Visualizations** — 3D interactive plots with Plotly
- [ ] **A/B Testing Framework** — measure campaign effectiveness

---

## 📊 Sample Output

**Cluster Distribution:**
```
Cluster 0: 1,234 customers (High-Value)
Cluster 1: 2,456 customers (Budget Shoppers)
Cluster 2: 987 customers (Young Professionals)
Cluster 3: 543 customers (Inactive)
```

**PCA Visualization:**
- 2D scatter plot showing clear separation between customer segments
- Color-coded clusters with centroids marked

---

## 👨‍💻 About the Author

**Nikhil More**
B.Tech CSE (AI/ML) — University of Mumbai (2023–2027)

- 🔗 [LinkedIn](https://www.linkedin.com/in/nikhil-moretech)
- 🐙 [GitHub](https://github.com/Khiladi-786)
- 📧 morenikhil7822@gmail.com

*Building ML solutions for real-world business problems.*

---

## 📄 License

This project is licensed under the MIT License — free for educational and commercial use.

---

## 🙏 Acknowledgments

- Dataset inspiration: [Kaggle Customer Segmentation](https://www.kaggle.com/)
- Streamlit documentation for dashboard design
- Scikit-learn for ML algorithms

---

⭐ **If you found this project useful, please give it a star!** ⭐

**🔗 Live Demo:** [Customer Segmentation Dashboard](https://your-streamlit-link.streamlit.app)
