import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv('data/Sales Dataset.csv')
    return df

@st.cache_data
def preprocess_data(df):
    """Feature Engineering & Data Preprocessing"""
    # Convert date columns
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Year-Month'] = pd.to_datetime(df['Year-Month'])
    
    # Date features
    df['Order_Year'] = df['Order Date'].dt.year
    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_Month_Name'] = df['Order Date'].dt.strftime('%B')
    df['Order_Day'] = df['Order Date'].dt.day
    df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek
    df['Order_DayName'] = df['Order Date'].dt.strftime('%A')
    df['Order_Quarter'] = df['Order Date'].dt.quarter
    df['Order_WeekOfYear'] = df['Order Date'].dt.isocalendar().week
    
    # Feature Engineering
    df['Profit_Margin'] = (df['Profit'] / df['Amount']) * 100
    df['Avg_Price_Per_Unit'] = df['Amount'] / df['Quantity']
    
    # Sales per Category (aggregated feature)
    category_sales = df.groupby('Category')['Amount'].mean()
    df['Category_Avg_Sales'] = df['Category'].map(category_sales)
    
    # Monthly trend (normalized)
    monthly_avg = df.groupby('Year-Month')['Amount'].mean()
    df['Monthly_Trend'] = df['Year-Month'].map(monthly_avg)
    
    # Is weekend
    df['Is_Weekend'] = df['Order_DayOfWeek'].isin([5, 6]).astype(int)
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Order_Month'].apply(get_season)
    
    # Encode categorical variables for ML
    le_category = LabelEncoder()
    le_subcategory = LabelEncoder()
    le_payment = LabelEncoder()
    le_state = LabelEncoder()
    le_city = LabelEncoder()
    le_month_name = LabelEncoder()
    le_day_name = LabelEncoder()
    le_season = LabelEncoder()
    
    df['Category_Encoded'] = le_category.fit_transform(df['Category'])
    df['SubCategory_Encoded'] = le_subcategory.fit_transform(df['Sub-Category'])
    df['Payment_Encoded'] = le_payment.fit_transform(df['PaymentMode'])
    df['State_Encoded'] = le_state.fit_transform(df['State'])
    df['City_Encoded'] = le_city.fit_transform(df['City'])
    df['Month_Name_Encoded'] = le_month_name.fit_transform(df['Order_Month_Name'])
    df['Day_Name_Encoded'] = le_day_name.fit_transform(df['Order_DayName'])
    df['Season_Encoded'] = le_season.fit_transform(df['Season'])
    
    return df

def time_series_analysis(df):
    """Time Series Analysis: Trends and Seasonality"""
    # Monthly sales aggregation
    monthly_sales = df.groupby('Year-Month').agg({
        'Amount': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    monthly_sales = monthly_sales.sort_values('Year-Month')
    monthly_sales.columns = ['Year-Month', 'Total_Sales', 'Total_Profit', 'Order_Count']
    
    # Calculate moving average
    monthly_sales['Moving_Avg_3M'] = monthly_sales['Total_Sales'].rolling(window=3, min_periods=1).mean()
    monthly_sales['Moving_Avg_6M'] = monthly_sales['Total_Sales'].rolling(window=6, min_periods=1).mean()
    
    # Seasonality: Average sales by month across all years
    seasonality = df.groupby('Order_Month_Name').agg({
        'Amount': 'mean',
        'Profit': 'mean'
    }).reset_index()
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_num = {m: i+1 for i, m in enumerate(month_order)}
    seasonality['Month_Num'] = seasonality['Order_Month_Name'].map(month_num)
    seasonality = seasonality.sort_values('Month_Num')
    
    return monthly_sales, seasonality

def train_ml_models(df):
    """Train Random Forest models for profit and sales prediction"""
    # Prepare features
    feature_cols = ['Quantity', 'Category_Encoded', 'SubCategory_Encoded', 
                    'Payment_Encoded', 'State_Encoded', 'City_Encoded',
                    'Order_Year', 'Order_Month', 'Order_Day', 'Order_DayOfWeek',
                    'Order_Quarter', 'Is_Weekend', 'Season_Encoded',
                    'Category_Avg_Sales', 'Monthly_Trend']
    
    # Model 1: Predict Profit
    X_profit = df[feature_cols].values
    y_profit = df['Profit'].values
    
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_profit, y_profit, test_size=0.2, random_state=42
    )
    
    rf_profit = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_profit.fit(X_train_p, y_train_p)
    
    profit_pred_train = rf_profit.predict(X_train_p)
    profit_pred_test = rf_profit.predict(X_test_p)
    
    profit_metrics = {
        'train_mse': mean_squared_error(y_train_p, profit_pred_train),
        'test_mse': mean_squared_error(y_test_p, profit_pred_test),
        'train_r2': r2_score(y_train_p, profit_pred_train),
        'test_r2': r2_score(y_test_p, profit_pred_test),
        'train_mae': mean_absolute_error(y_train_p, profit_pred_train),
        'test_mae': mean_absolute_error(y_test_p, profit_pred_test)
    }
    
    # Model 2: Predict Amount (Sales)
    X_sales = df[feature_cols].values
    y_sales = df['Amount'].values
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_sales, y_sales, test_size=0.2, random_state=42
    )
    
    rf_sales = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_sales.fit(X_train_s, y_train_s)
    
    sales_pred_train = rf_sales.predict(X_train_s)
    sales_pred_test = rf_sales.predict(X_test_s)
    
    sales_metrics = {
        'train_mse': mean_squared_error(y_train_s, sales_pred_train),
        'test_mse': mean_squared_error(y_test_s, sales_pred_test),
        'train_r2': r2_score(y_train_s, sales_pred_train),
        'test_r2': r2_score(y_test_s, sales_pred_test),
        'train_mae': mean_absolute_error(y_train_s, sales_pred_train),
        'test_mae': mean_absolute_error(y_test_s, sales_pred_test)
    }
    
    # Feature importance
    feature_importance_profit = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_profit.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance_sales = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_sales.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return (rf_profit, rf_sales, profit_metrics, sales_metrics, 
            feature_importance_profit, feature_importance_sales)

def predict_with_model(model, df, feature_cols, input_data):
    """Make prediction with user input"""
    input_df = pd.DataFrame([input_data])
    # Encode categorical inputs
    input_df['Category_Encoded'] = df[df['Category'] == input_data['Category']]['Category_Encoded'].iloc[0] if input_data['Category'] in df['Category'].values else 0
    input_df['SubCategory_Encoded'] = df[df['Sub-Category'] == input_data['Sub-Category']]['SubCategory_Encoded'].iloc[0] if input_data['Sub-Category'] in df['Sub-Category'].values else 0
    input_df['Payment_Encoded'] = df[df['PaymentMode'] == input_data['PaymentMode']]['Payment_Encoded'].iloc[0] if input_data['PaymentMode'] in df['PaymentMode'].values else 0
    input_df['State_Encoded'] = df[df['State'] == input_data['State']]['State_Encoded'].iloc[0] if input_data['State'] in df['State'].values else 0
    input_df['City_Encoded'] = df[df['City'] == input_data['City']]['City_Encoded'].iloc[0] if input_data['City'] in df['City'].values else 0
    input_df['Month_Name_Encoded'] = df[df['Order_Month_Name'] == input_data['Order_Month_Name']]['Month_Name_Encoded'].iloc[0] if input_data['Order_Month_Name'] in df['Order_Month_Name'].values else 0
    input_df['Day_Name_Encoded'] = df[df['Order_DayName'] == input_data['Order_DayName']]['Day_Name_Encoded'].iloc[0] if input_data['Order_DayName'] in df['Order_DayName'].values else 0
    input_df['Season_Encoded'] = df[df['Season'] == input_data['Season']]['Season_Encoded'].iloc[0] if input_data['Season'] in df['Season'].values else 0
    
    input_df['Category_Avg_Sales'] = df.groupby('Category')['Amount'].mean().get(input_data['Category'], df['Amount'].mean())
    input_df['Monthly_Trend'] = df['Monthly_Trend'].mean()
    
    X = input_df[feature_cols].values
    prediction = model.predict(X)[0]
    
    return prediction

def main():
    # Load and prepare data
    with st.spinner('Loading data...'):
        df_raw = load_data()
        df = preprocess_data(df_raw.copy())
        monthly_sales, seasonality = time_series_analysis(df)
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.radio("Go to:", [
        "📊 Dashboard",
        "📈 Time Series",
        "🤖 Machine Learning",
        "🔮 Prediction",
        "📉 EDA"
    ])
    
    # Header
    st.markdown('<div class="main-header">📊 Sales Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive Sales Analysis with Machine Learning & Time Series</div>', unsafe_allow_html=True)
    
    if page == "📊 Dashboard":
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("💰 Total Revenue", f"${df['Amount'].sum():,.0f}", 
                     f"{((df['Amount'].sum() - df['Amount'].mean() * len(df)) / (df['Amount'].mean() * len(df)) * 100):.1f}%")
        
        with col2:
            st.metric("📈 Total Profit", f"${df['Profit'].sum():,.0f}",
                     f"{(df['Profit'].sum() / df['Amount'].sum() * 100):.1f}% margin")
        
        with col3:
            st.metric("🛒 Total Orders", f"{len(df):,}",
                     f"{df['Order ID'].nunique():,} unique")
        
        with col4:
            st.metric("📦 Products Sold", f"{df['Quantity'].sum():,}",
                     f"{df['Quantity'].mean():.0f} avg/order")
        
        with col5:
            st.metric("👥 Customers", f"{df['CustomerName'].nunique():,}",
                     f"{len(df) / df['CustomerName'].nunique():.1f} orders each")
        
        st.markdown("---")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df.groupby('Category')['Amount'].sum().reset_index(),
                x='Category', y='Amount',
                title='Total Sales by Category',
                color='Amount',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=df['PaymentMode'].value_counts().values,
                names=df['PaymentMode'].value_counts().index,
                title='Payment Mode Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            top_states = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(
                top_states,
                x='State', y='Amount',
                title='Top 10 States by Sales',
                color='Amount',
                color_continuous_scale='Hot'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_customers = df.groupby('CustomerName')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(
                top_customers,
                y='CustomerName', x='Amount',
                title='Top 10 Customers by Spending',
                color='Amount',
                color_continuous_scale='Blues',
                orientation='h'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts row 3
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.box(
                df,
                x='Category', y='Profit_Margin',
                title='Profit Margin by Category',
                color='Category'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            subcat_sales = df.groupby('Sub-Category')['Amount'].sum().sort_values(ascending=True).tail(10).reset_index()
            fig = px.bar(
                subcat_sales,
                y='Sub-Category', x='Amount',
                title='Top 10 Sub-Categories',
                color='Amount',
                color_continuous_scale='Greens',
                orientation='h'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            daily_sales = df.groupby('Order_DayName')['Amount'].sum()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_sales = daily_sales.reindex(day_order).reset_index()
            fig = px.line(
                daily_sales,
                x='Order_DayName', y='Amount',
                title='Sales by Day of Week',
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Time Series":
        st.header("📈 Time Series Analysis")
        
        # Sales Trend
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Sales Trend', 'Seasonality Pattern'),
            vertical_spacing=0.15
        )
        
        # Trend line
        fig.add_trace(
            go.Scatter(
                x=monthly_sales['Year-Month'],
                y=monthly_sales['Total_Sales'],
                mode='lines+markers',
                name='Monthly Sales',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_sales['Year-Month'],
                y=monthly_sales['Moving_Avg_3M'],
                mode='lines',
                name='3-Month MA',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Seasonality
        fig.add_trace(
            go.Bar(
                x=seasonality['Order_Month_Name'],
                y=seasonality['Amount'],
                name='Avg Sales',
                marker_color='teal'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional time series metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.line(
                monthly_sales,
                x='Year-Month', y='Total_Profit',
                title='Monthly Profit Trend',
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                monthly_sales,
                x='Year-Month', y='Order_Count',
                title='Monthly Order Count',
                color='Order_Count',
                color_continuous_scale='Purples'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            yearly_season = df.groupby(['Order_Year', 'Season'])['Amount'].sum().reset_index()
            fig = px.bar(
                yearly_season,
                x='Order_Year', y='Amount',
                color='Season',
                title='Sales by Year & Season',
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-Year comparison
        st.subheader("📊 Year-over-Year Monthly Comparison")
        yoy_data = df.groupby(['Order_Year', 'Order_Month'])['Amount'].sum().reset_index()
        fig = px.line(
            yoy_data,
            x='Order_Month', y='Amount',
            color='Order_Year',
            title='Monthly Sales by Year',
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Machine Learning":
        st.header("🤖 Machine Learning - Random Forest Models")
        
        with st.spinner('Training models...'):
            (rf_profit, rf_sales, profit_metrics, sales_metrics, 
             feat_imp_profit, feat_imp_sales) = train_ml_models(df)
        
        # Model Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Profit Prediction Model")
            st.markdown("**Model Performance:**")
            metrics_df = pd.DataFrame({
                'Metric': ['Train MSE', 'Test MSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE'],
                'Value': [
                    f"{profit_metrics['train_mse']:,.2f}",
                    f"{profit_metrics['test_mse']:,.2f}",
                    f"{profit_metrics['train_r2']:.4f}",
                    f"{profit_metrics['test_r2']:.4f}",
                    f"{profit_metrics['train_mae']:,.2f}",
                    f"{profit_metrics['test_mae']:,.2f}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            st.success(f"✅ Test R² Score: {profit_metrics['test_r2']:.4f}")
        
        with col2:
            st.subheader("📊 Sales Prediction Model")
            st.markdown("**Model Performance:**")
            metrics_df = pd.DataFrame({
                'Metric': ['Train MSE', 'Test MSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE'],
                'Value': [
                    f"{sales_metrics['train_mse']:,.2f}",
                    f"{sales_metrics['test_mse']:,.2f}",
                    f"{sales_metrics['train_r2']:.4f}",
                    f"{sales_metrics['test_r2']:.4f}",
                    f"{sales_metrics['train_mae']:,.2f}",
                    f"{sales_metrics['test_mae']:,.2f}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            st.success(f"✅ Test R² Score: {sales_metrics['test_r2']:.4f}")
        
        st.markdown("---")
        
        # Feature Importance
        st.subheader("🔍 Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                feat_imp_profit.head(10),
                y='Feature', x='Importance',
                title='Top 10 Features - Profit Prediction',
                orientation='h',
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                feat_imp_sales.head(10),
                y='Feature', x='Importance',
                title='Top 10 Features - Sales Prediction',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Architecture Info
        st.subheader("⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Profit Prediction Model:**
            - Algorithm: Random Forest Regressor
            - Estimators: 100
            - Max Depth: 15
            - Min Samples Split: 5
            - Min Samples Leaf: 2
            """)
        
        with col2:
            st.info("""
            **Sales Prediction Model:**
            - Algorithm: Random Forest Regressor
            - Estimators: 100
            - Max Depth: 15
            - Min Samples Split: 5
            - Min Samples Leaf: 2
            """)
    
    elif page == "🔮 Prediction":
        st.header("🔮 Sales & Profit Predictor")
        st.markdown("Enter order details to predict profit and sales amount")
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category = st.selectbox("Category", df['Category'].unique())
            subcategory = st.selectbox("Sub-Category", 
                                      df[df['Category'] == category]['Sub-Category'].unique())
            quantity = st.slider("Quantity", 1, 20, 10)
        
        with col2:
            payment = st.selectbox("Payment Mode", df['PaymentMode'].unique())
            state = st.selectbox("State", df['State'].unique())
            city = st.selectbox("City", 
                               df[df['State'] == state]['City'].unique() if len(df[df['State'] == state]) > 0 else df['City'].unique())
        
        with col3:
            month = st.selectbox("Month", range(1, 13))
            day = st.slider("Day", 1, 31, 15)
            year = st.selectbox("Year", sorted(df['Order_Year'].unique()))
        
        # Get derived features
        month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B')
        day_name = pd.Timestamp(year=year, month=month, day=day).strftime('%A')
        day_of_week = pd.Timestamp(year=year, month=month, day=day).dayofweek
        quarter = pd.Timestamp(year=year, month=month, day=day).quarter
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        season = get_season(month)
        
        input_data = {
            'Quantity': quantity,
            'Category': category,
            'Sub-Category': subcategory,
            'PaymentMode': payment,
            'State': state,
            'City': city,
            'Order_Year': year,
            'Order_Month': month,
            'Order_Day': day,
            'Order_DayOfWeek': day_of_week,
            'Order_Quarter': quarter,
            'Is_Weekend': 1 if day_of_week >= 5 else 0,
            'Order_Month_Name': month_name,
            'Order_DayName': day_name,
            'Season': season,
            'Category_Avg_Sales': df[df['Category'] == category]['Amount'].mean(),
            'Monthly_Trend': df['Monthly_Trend'].mean()
        }
        
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            with st.spinner('Making predictions...'):
                (rf_profit, rf_sales, _, _, _, _) = train_ml_models(df)
                
                feature_cols = ['Quantity', 'Category_Encoded', 'SubCategory_Encoded', 
                                'Payment_Encoded', 'State_Encoded', 'City_Encoded',
                                'Order_Year', 'Order_Month', 'Order_Day', 'Order_DayOfWeek',
                                'Order_Quarter', 'Is_Weekend', 'Season_Encoded',
                                'Category_Avg_Sales', 'Monthly_Trend']
                
                # Encode inputs
                input_encoded = input_data.copy()
                input_encoded['Category_Encoded'] = df[df['Category'] == category]['Category_Encoded'].iloc[0]
                input_encoded['SubCategory_Encoded'] = df[df['Sub-Category'] == subcategory]['SubCategory_Encoded'].iloc[0]
                input_encoded['Payment_Encoded'] = df[df['PaymentMode'] == payment]['Payment_Encoded'].iloc[0]
                input_encoded['State_Encoded'] = df[df['State'] == state]['State_Encoded'].iloc[0]
                input_encoded['City_Encoded'] = df[df['City'] == city]['City_Encoded'].iloc[0]
                input_encoded['Month_Name_Encoded'] = df[df['Order_Month_Name'] == month_name]['Month_Name_Encoded'].iloc[0]
                input_encoded['Day_Name_Encoded'] = df[df['Order_DayName'] == day_name]['Day_Name_Encoded'].iloc[0]
                input_encoded['Season_Encoded'] = df[df['Season'] == season]['Season_Encoded'].iloc[0]
                
                input_df = pd.DataFrame([input_encoded])
                X = input_df[feature_cols].values
                
                predicted_profit = rf_profit.predict(X)[0]
                predicted_amount = rf_sales.predict(X)[0]
                
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💰 Predicted Sales Amount", f"${predicted_amount:,.2f}")
                
                with col2:
                    st.metric("📈 Predicted Profit", f"${predicted_profit:,.2f}")
                
                with col3:
                    profit_margin = (predicted_profit / predicted_amount) * 100 if predicted_amount > 0 else 0
                    st.metric("📊 Predicted Profit Margin", f"{profit_margin:.2f}%")
    
    elif page == "📉 EDA":
        st.header("📉 Exploratory Data Analysis")
        
        # Data overview
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Date Range", f"{df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
        
        st.dataframe(df.head(), use_container_width=True)
        
        # Distribution analysis
        st.subheader("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, x='Amount', nbins=50,
                title='Amount Distribution',
                marginal='box'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, x='Profit', nbins=50,
                title='Profit Distribution',
                marginal='box',
                color_discrete_sequence=['green']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Matrix")
        corr_cols = ['Amount', 'Profit', 'Quantity', 'Profit_Margin', 'Avg_Price_Per_Unit']
        corr_matrix = df[corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Correlation Heatmap'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category analysis
        st.subheader("🏷️ Category Deep Dive")
        
        cat_stats = df.groupby('Category').agg({
            'Amount': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Quantity': 'sum'
        }).round(2)
        cat_stats.columns = ['Total Sales', 'Avg Sales', 'Total Profit', 'Avg Profit', 'Total Quantity']
        st.dataframe(cat_stats, use_container_width=True)
        
        # Geographic analysis
        st.subheader("🌍 Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            state_data = df.groupby('State')['Amount'].sum().reset_index()
            fig = px.bar(
                state_data.sort_values('Amount', ascending=False).head(15),
                x='State', y='Amount',
                title='Sales by State',
                color='Amount',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            city_data = df.groupby('City')['Amount'].sum().reset_index()
            fig = px.bar(
                city_data.sort_values('Amount', ascending=False).head(15),
                y='City', x='Amount',
                title='Top 15 Cities by Sales',
                color='Amount',
                color_continuous_scale='Plasma',
                orientation='h'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
