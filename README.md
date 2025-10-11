# 🛍️ Mall Customer Segmentation

## Overview

Customer segmentation project using K-Means clustering to identify 5 distinct customer groups based on income and spending behavior.

## Quick Start

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit
```

### Run Project

```bash
# Step 1: Open and run Data_Exploration.ipynb
# Step 2: Open and run Model_Training.ipynb
# Step 3: Launch dashboard
streamlit run app.py
```

## Project Structure

```
├── Mall_Customers.csv                 # Original data (200 customers)
├── Data_Exploration.ipynb             # EDA notebook
├── Model_Training.ipynb               # Clustering (K=5)
├── app.py                             # Streamlit dashboard
├── Mall_Customers_Clustered.csv       # Output with clusters
└── README.md
```

## Why K=5?

| K     | Silhouette | Davies-Bouldin | Status      |
| ----- | ---------- | -------------- | ----------- |
| 4     | 0.494      | 0.710          | Good        |
| **5** | **0.555**  | **0.572**      | **✅ Best** |
| 6     | 0.540      | 0.655          | Declining   |

**K=5 has the highest Silhouette Score (0.555) and lowest Davies-Bouldin Index (0.572)**

## Customer Segments

### Cluster 0: Low Income, Low Spenders 💼

- **Profile:** Budget-conscious customers
- **Strategy:** Discounts, value offerings

### Cluster 1: High Income, Low Spenders 💎

- **Profile:** Wealthy but selective
- **Strategy:** Premium quality products

### Cluster 2: Low Income, High Spenders 🎯

- **Profile:** Enthusiastic shoppers
- **Strategy:** Payment plans, trendy items

### Cluster 3: High Income, High Spenders ⭐

- **Profile:** VIP customers (most valuable)
- **Strategy:** Luxury items, VIP programs

### Cluster 4: Moderate Spenders 🌟

- **Profile:** Balanced middle-market
- **Strategy:** Seasonal promotions

## Dashboard Pages

1. **Overview** - Key metrics and statistics
2. **Data Exploration** - EDA visualizations
3. **Clustering Results** - 5 cluster visualization
4. **Customer Insights** - Segment profiles & strategies
5. **Predict Cluster** - Classify new customers

## Key Results

- **200 customers** segmented into **5 clusters**
- **Silhouette Score:** 0.555 (good separation)
- **Income & Spending:** Weakly correlated (good for clustering)
- **20% are high-value customers** (Cluster 3)

## Technologies

- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit (Dashboard)

## Task Completion

✅ Dataset: Mall Customer (Kaggle)  
✅ Clustering by income & spending  
✅ Feature scaling  
✅ K-Means with optimal K determination  
✅ 2D visualizations  
✅ BONUS: DBSCAN algorithm  
✅ BONUS: Average spending analysis

---

**Ready to use!** Open the notebooks in order, then launch the Streamlit app.
