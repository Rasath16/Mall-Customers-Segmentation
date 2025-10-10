import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load clustered data from model training notebook"""
    try:
        df = pd.read_csv('data/Mall_Customers_Clustered.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please run the Model Training Notebook first to generate 'Mall_Customers_Clustered.csv'")
        st.stop()

@st.cache_resource
def load_models():
    """Load and train models for predictions"""
    try:
        df = pd.read_csv('data/Mall_Customers_Clustered.csv')
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train K-Means with K=5 (optimal based on Silhouette Score)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        return kmeans, scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Load data and models
df = load_data()
kmeans, scaler = load_models()

# Color palette for 5 clusters
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Overview", "📊 Data Exploration", "🎯 Clustering Results", 
     "💡 Customer Insights", "🔮 Predict Cluster"]
)

st.sidebar.markdown("---")
st.sidebar.title("📈 Dataset Statistics")
st.sidebar.metric("Total Customers", len(df))
st.sidebar.metric("Number of Clusters", 5)
st.sidebar.metric("Optimal K", "5 (Silhouette: 0.555)")

st.sidebar.markdown("---")
st.sidebar.info("""
**About this App:**

This dashboard provides insights into customer segmentation using K-Means clustering (K=5).

**Model Selection:**
- Silhouette Score: 0.555
- Davies-Bouldin: 0.572
- Best performing K value
""")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.markdown('<p class="main-header">🛍️ Mall Customer Segmentation Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Customers",
            value=len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            label="👥 Average Age",
            value=f"{df['Age'].mean():.1f} years",
            delta=None
        )
    
    with col3:
        st.metric(
            label="💰 Average Income",
            value=f"${df['Annual Income (k$)'].mean():.0f}k",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🛒 Average Spending",
            value=f"{df['Spending Score (1-100)'].mean():.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(
            df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 
                'Spending Score (1-100)', 'Cluster']].head(10),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.dataframe(
            df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe(),
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # Project description
    st.subheader("🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Objective
        This project segments mall customers into **5 distinct groups** using **K-Means clustering** 
        to enable targeted marketing strategies.
        
        ### Features Used
        - **Annual Income (k$)**: Customer's yearly income
        - **Spending Score (1-100)**: Behavioral spending score
        
        ### Why K=5?
        After evaluating K=2 to K=10:
        - **Highest Silhouette Score: 0.555**
        - **Lowest Davies-Bouldin: 0.572**
        - Best balance of cluster quality
        """)
    
    with col2:
        st.markdown("""
        ### Methodology
        1. **Data Preprocessing**: Feature scaling using StandardScaler
        2. **Cluster Optimization**: Elbow method, Silhouette Score, Davies-Bouldin Index
        3. **Model Training**: K-Means clustering with K=5
        4. **Evaluation**: Multiple metrics and business insights
        
        ### Results
        ✅ **5 distinct customer segments** identified  
        ✅ Clear separation between clusters  
        ✅ Actionable marketing strategies  
        ✅ BONUS: DBSCAN comparison analysis  
        """)

# ============================================================================
# PAGE 2: DATA EXPLORATION
# ============================================================================

elif page == "📊 Data Exploration":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    # Distribution plots
    st.subheader("📈 Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Age Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Annual Income (k$)'], bins=20, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Annual Income (k$)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Income Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Spending Score (1-100)'], bins=20, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Spending Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Spending Score Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        gender_counts = df['Gender'].value_counts()
        ax.bar(gender_counts.index, gender_counts.values, 
               color=['#FF69B4', '#4169E1'], edgecolor='black')
        ax.set_xlabel('Gender', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Gender Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Interactive scatter plot
    st.subheader("🔍 Income vs Spending Score Analysis")
    
    fig = px.scatter(
        df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Age',
        hover_data=['Gender', 'Age', 'CustomerID'],
        color_continuous_scale='Viridis',
        title='Income vs Spending Score (colored by Age)',
        labels={'Annual Income (k$)': 'Income (k$)', 
                'Spending Score (1-100)': 'Spending Score'}
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=1, linecolor='black',
                   cbar_kws={'label': 'Correlation'}, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.info("""
        **Key Insights:**
        
        - Income and Spending Score have **weak correlation**
        - This confirms they are relatively independent features
        - Perfect for clustering analysis
        - Age has minimal impact on spending patterns
        """)

# ============================================================================
# PAGE 3: CLUSTERING RESULTS
# ============================================================================

elif page == "🎯 Clustering Results":
    st.title("🎯 K-Means Clustering Results (K=5)")
    st.markdown("---")
    
    # Cluster metrics
    st.subheader("📊 Cluster Distribution")
    cols = st.columns(5)
    
    for i in range(5):
        count = len(df[df['Cluster'] == i])
        percentage = count / len(df) * 100
        
        with cols[i]:
            st.metric(
                label=f"Cluster {i}",
                value=f"{count}",
                delta=f"{percentage:.1f}%"
            )
    
    st.markdown("---")
    
    # Cluster visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Cluster Visualization")
        
        fig = px.scatter(
            df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            color='Cluster',
            hover_data=['Age', 'Gender', 'CustomerID'],
            color_discrete_sequence=colors,
            title='Customer Segments: Income vs Spending Score',
            labels={'Annual Income (k$)': 'Income (k$)',
                   'Spending Score (1-100)': 'Spending Score',
                   'Cluster': 'Cluster'}
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Cluster Sizes")
        
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(cluster_counts.index, cluster_counts.values,
              color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Customers', fontsize=11)
        ax.set_title('Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(range(5))
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(cluster_counts.values):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Cluster characteristics
    st.subheader("📋 Cluster Characteristics")
    
    cluster_profile = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 
                                              'Spending Score (1-100)']].mean().round(2)
    
    # Add cluster names
    cluster_names = {
        0: "Low Income, Low Spenders",
        1: "High Income, Low Spenders",
        2: "Low Income, High Spenders",
        3: "High Income, High Spenders",
        4: "Moderate Income, Moderate Spenders"
    }
    
    cluster_profile['Segment Name'] = cluster_profile.index.map(cluster_names)
    cluster_profile = cluster_profile[['Segment Name', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    st.dataframe(cluster_profile, use_container_width=True)
    
    st.markdown("---")
    
    # Box plots
    st.subheader("📦 Feature Distribution by Cluster")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        df.boxplot(column=col, by='Cluster', ax=axes[idx])
        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Cluster')
        axes[idx].set_ylabel(col)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Model Performance
    st.subheader("🎯 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Silhouette Score", "0.555", help="Range: -1 to 1, Higher is better")
    with col2:
        st.metric("Davies-Bouldin Index", "0.572", help="Lower is better")
    with col3:
        st.metric("Optimal K", "5", help="Best performing cluster count")

# ============================================================================
# PAGE 4: CUSTOMER INSIGHTS
# ============================================================================

elif page == "💡 Customer Insights":
    st.title("💡 Customer Insights & Profiles")
    st.markdown("---")
    
    # Average spending by cluster
    st.subheader("💰 Average Spending Score by Cluster")
    
    avg_spending = df.groupby('Cluster')['Spending Score (1-100)'].mean().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(5), avg_spending.values, color=colors,
                  edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Average Spending Score', fontsize=12)
    ax.set_title('Average Spending Score by Cluster', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_spending.values):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold', fontsize=12)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Cluster selection
    st.subheader("🔍 Detailed Cluster Analysis")
    
    selected_cluster = st.selectbox("Select a cluster for detailed analysis:", 
                                    list(range(5)),
                                    format_func=lambda x: f"Cluster {x}")
    
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cluster Size", len(cluster_data))
    with col2:
        st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f} years")
    with col3:
        st.metric("Avg Income", f"${cluster_data['Annual Income (k$)'].mean():.0f}k")
    with col4:
        st.metric("Avg Spending", f"{cluster_data['Spending Score (1-100)'].mean():.1f}")
    
    st.markdown("---")
    
    # Detailed visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📍 Cluster {selected_cluster} - Customer Distribution")
        
        fig = px.scatter(
            cluster_data,
            x='Age',
            y='Annual Income (k$)',
            color='Spending Score (1-100)',
            hover_data=['Gender', 'CustomerID'],
            color_continuous_scale='Viridis',
            title=f'Cluster {selected_cluster}: Age vs Income (colored by Spending)'
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"📊 Cluster {selected_cluster} - Statistics")
        summary = cluster_data[['Age', 'Annual Income (k$)', 
                                'Spending Score (1-100)']].describe().round(2)
        st.dataframe(summary, use_container_width=True, height=350)
    
    st.markdown("---")
    
    # Customer list
    st.subheader(f"👥 Cluster {selected_cluster} - Customer List")
    st.dataframe(
        cluster_data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 
                     'Spending Score (1-100)']].head(20),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Cluster interpretation
    st.subheader("📌 Cluster Interpretation & Marketing Strategy")
    
    interpretations = {
        0: {
            "name": "💼 Low Income, Low Spenders",
            "icon": "💼",
            "description": "Budget-conscious customers with limited purchasing power. They are price-sensitive and look for value.",
            "characteristics": [
                "Lower annual income range ($15k-$40k)",
                "Conservative spending behavior (Score: 1-40)",
                "Price-sensitive segment",
                "Focus on necessities over luxuries"
            ],
            "strategy": [
                "Value-based product offerings",
                "Discount and loyalty programs",
                "Budget-friendly promotions",
                "Bulk purchase discounts"
            ],
            "color": "blue"
        },
        1: {
            "name": "💎 High Income, Low Spenders",
            "icon": "💎",
            "description": "Wealthy but careful customers with high savings tendency. They prioritize quality over quantity.",
            "characteristics": [
                "High income levels ($60k-$140k)",
                "Selective purchasing behavior (Score: 1-40)",
                "Quality over quantity mindset",
                "Research before buying"
            ],
            "strategy": [
                "Premium quality products",
                "Investment-focused offerings",
                "Exclusive membership benefits",
                "Long-term value propositions"
            ],
            "color": "green"
        },
        2: {
            "name": "🎯 Low Income, High Spenders",
            "icon": "🎯",
            "description": "Enthusiastic buyers despite limited income. They are impulse buyers who love shopping.",
            "characteristics": [
                "Lower income range ($15k-$40k)",
                "High spending frequency (Score: 60-100)",
                "Impulse buying tendencies",
                "Trend-focused consumers"
            ],
            "strategy": [
                "Flexible payment plans",
                "Affordable trendy items",
                "Installment options (Buy Now Pay Later)",
                "Limited-time offers"
            ],
            "color": "orange"
        },
        3: {
            "name": "⭐ High Income, High Spenders",
            "icon": "⭐",
            "description": "Premium customers with strong purchasing power. The most valuable segment for the business.",
            "characteristics": [
                "High disposable income ($60k-$140k)",
                "Frequent high-value purchases (Score: 60-100)",
                "Brand loyal customers",
                "Seek premium experiences"
            ],
            "strategy": [
                "Luxury product lines",
                "VIP treatment programs",
                "Exclusive early access deals",
                "Personalized shopping experiences"
            ],
            "color": "red"
        },
        4: {
            "name": "🌟 Moderate Income, Moderate Spenders",
            "icon": "🌟",
            "description": "Balanced customers with moderate income and spending. They represent the middle market segment.",
            "characteristics": [
                "Moderate income range ($40k-$70k)",
                "Balanced spending (Score: 40-60)",
                "Pragmatic decision makers",
                "Value quality and price balance"
            ],
            "strategy": [
                "Seasonal promotions",
                "Loyalty rewards programs",
                "Mid-tier product offerings",
                "Bundle deals and packages"
            ],
            "color": "purple"
        }
    }
    
    info = interpretations[selected_cluster]
    
    # Display in columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"### {info['icon']} {info['name']}")
        st.markdown(f"**Profile Description:**")
        st.info(info['description'])
        
        st.markdown("**Key Characteristics:**")
        for char in info['characteristics']:
            st.markdown(f"• {char}")
    
    with col2:
        st.markdown("### 📈 Marketing Strategy")
        st.success(f"**Target Size:** {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
        
        st.markdown("**Recommended Approaches:**")
        for strat in info['strategy']:
            st.markdown(f"✓ {strat}")
    
    st.markdown("---")
    
    # Additional insights
    st.subheader("📊 Additional Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_dist = cluster_data['Gender'].value_counts()
        st.metric("Male Customers", gender_dist.get('Male', 0))
        st.metric("Female Customers", gender_dist.get('Female', 0))
    
    with col2:
        age_range = f"{cluster_data['Age'].min():.0f} - {cluster_data['Age'].max():.0f}"
        st.metric("Age Range", age_range)
        st.metric("Median Age", f"{cluster_data['Age'].median():.0f} years")
    
    with col3:
        income_range = f"${cluster_data['Annual Income (k$)'].min():.0f}k - ${cluster_data['Annual Income (k$)'].max():.0f}k"
        st.metric("Income Range", income_range)
        st.metric("Median Income", f"${cluster_data['Annual Income (k$)'].median():.0f}k")

# ============================================================================
# PAGE 5: PREDICT CLUSTER
# ============================================================================

elif page == "🔮 Predict Cluster":
    st.title("🔮 Predict Customer Cluster")
    st.markdown("---")
    
    st.markdown("""
    ### Enter Customer Information
    Input the customer's annual income and spending score to predict which cluster they belong to.
    This tool helps you instantly categorize new customers for targeted marketing.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.slider(
            "💰 Annual Income (k$)",
            min_value=15,
            max_value=140,
            value=60,
            step=1,
            help="Customer's yearly income in thousands of dollars"
        )
        
        st.info(f"**Selected Income:** ${income},000")
    
    with col2:
        spending = st.slider(
            "🛒 Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Behavioral score based on customer spending patterns"
        )
        
        st.info(f"**Selected Score:** {spending}/100")
    
    st.markdown("---")
    
    if st.button("🎯 Predict Cluster", type="primary", use_container_width=True):
        # Prepare input
        input_data = np.array([[income, spending]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict cluster
        predicted_cluster = kmeans.predict(input_scaled)[0]
        
        st.markdown("---")
        st.success(f"### 🎉 Predicted Cluster: **{predicted_cluster}**")
        
        # Show cluster info
        cluster_info = {
            0: {
                "name": "💼 Low Income, Low Spenders",
                "profile": "Budget-conscious customers",
                "strategy": "Value offerings, discount programs, loyalty rewards"
            },
            1: {
                "name": "💎 High Income, Low Spenders",
                "profile": "Wealthy but selective buyers",
                "strategy": "Premium quality products, exclusive memberships"
            },
            2: {
                "name": "🎯 Low Income, High Spenders",
                "profile": "Enthusiastic shoppers",
                "strategy": "Installment plans, affordable trendy items, BNPL options"
            },
            3: {
                "name": "⭐ High Income, High Spenders",
                "profile": "Premium VIP customers",
                "strategy": "Luxury items, VIP programs, personalized service"
            },
            4: {
                "name": "🌟 Moderate Income, Moderate Spenders",
                "profile": "Balanced middle-market segment",
                "strategy": "Seasonal promotions, bundle deals, loyalty programs"
            }
        }
        
        info = cluster_info[predicted_cluster]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            ### {info['name']}
            
            **Customer Profile:**  
            {info['profile']}
            
            **Input Values:**
            - Annual Income: ${income}k
            - Spending Score: {spending}/100
            """)
        
        with col2:
            st.success(f"""
            ### 📈 Recommended Marketing Strategy
            
            {info['strategy']}
            
            **Next Steps:**
            - Target with personalized offers
            - Assign to appropriate sales team
            - Apply segment-specific promotions
            """)
        
        # Visualize position
        st.markdown("---")
        st.subheader("📍 Customer Position in Cluster Space")
        
        # Create visualization with all clusters
        fig = px.scatter(
            df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            color='Cluster',
            color_discrete_sequence=colors,
            opacity=0.4,
            labels={'Annual Income (k$)': 'Income (k$)',
                   'Spending Score (1-100)': 'Spending Score',
                   'Cluster': 'Cluster'}
        )
        
        # Add the new customer point
        fig.add_scatter(
            x=[income],
            y=[spending],
            mode='markers+text',
            marker=dict(size=25, color='red', symbol='star',
                       line=dict(width=3, color='black')),
            text=['New Customer'],
            textposition='top center',
            textfont=dict(size=14, color='red', family='Arial Black'),
            name='New Customer',
            showlegend=True
        )
        
        fig.update_layout(
            height=600,
            title='Your Customer Position Among All Clusters',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show similar customers
        st.markdown("---")
        st.subheader(f"👥 Similar Customers in Cluster {predicted_cluster}")
        
        cluster_customers = df[df['Cluster'] == predicted_cluster].copy()
        
        # Calculate similarity (Euclidean distance)
        cluster_customers['Distance'] = np.sqrt(
            (cluster_customers['Annual Income (k$)'] - income)**2 + 
            (cluster_customers['Spending Score (1-100)'] - spending)**2
        )
        
        # Show top 10 most similar customers
        similar_customers = cluster_customers.nsmallest(10, 'Distance')[
            ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 
             'Spending Score (1-100)', 'Distance']
        ].round(2)
        
        st.dataframe(similar_customers, use_container_width=True)
        
        st.info("""
        **💡 Tip:** These are the 10 most similar customers based on income and spending patterns. 
        You can analyze their purchase history to predict what the new customer might be interested in!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Mall Customer Segmentation Dashboard</strong></p>
    <p style='font-size: 12px;'>Built with Streamlit • K-Means Clustering (K=5) • Scikit-learn</p>
    <p style='font-size: 11px;'>Optimal K selected: Silhouette Score = 0.555 | Davies-Bouldin = 0.572</p>
</div>
""", unsafe_allow_html=True)