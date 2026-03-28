# AI_Sales_Insights
AI Sales Insights Dashboard Project
# ==============================
# AI SALES INSIGHTS DASHBOARD
# ==============================
# End-to-end project using Streamlit
## Problem Statement

Sales teams often lack visibility into trends, future performance, and customer behavior, leading to poor decision-making.

## Objective
Build an AI-powered dashboard to:

Forecast future sales
Segment customers
Generate actionable business insights

##Solution Overview

This project uses Machine Learning and data visualization to:

Predict future sales using regression
Segment customers using clustering
Provide AI-driven recommendations

## Architecture

Data → Processing (Pandas) → ML Models → Streamlit Dashboard → Insights

## Tech Stack
Python (Pandas, NumPy, Scikit-learn)
Streamlit
Matplotlib

## Features

✔ Sales Forecasting (Next 10 Days)
✔ Customer Segmentation (KMeans Clustering)
✔ Interactive Dashboard
✔ AI-Based Business Insights

## Business Impact
Enables proactive decision-making
Identifies high-value customers
Improves sales strategy planning
▶️ How to Run
pip install pandas numpy scikit-learn streamlit matplotlib
streamlit run app.py
📸 Demo

(Add screenshots here after running)

## Future Enhancements
Integrate real-time CRM data
Add OpenAI-based insights
Deploy on cloud (AWS / Azure)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'sales': np.random.randint(100, 500, 100),
        'customer_id': np.random.randint(1, 50, 100)
    })
    return data

# ------------------------------
# FORECASTING FUNCTION
# ------------------------------
def forecast_sales(df):
    df['day'] = np.arange(len(df))
    X = df[['day']]
    y = df['sales']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(df), len(df)+10).reshape(-1,1)
    predictions = model.predict(future_days)

    return predictions

# ------------------------------
# CUSTOMER SEGMENTATION
# ------------------------------
def segment_customers(df):
    customer_data = df.groupby('customer_id')['sales'].sum().reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data['segment'] = kmeans.fit_predict(customer_data[['sales']])
    return customer_data

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("AI Sales Insights Dashboard")

# Load data
df = load_data()

st.subheader("Raw Sales Data")
st.write(df.head())

# Forecast
st.subheader("Sales Forecast (Next 10 Days)")
predictions = forecast_sales(df)
st.write(predictions)

# Plot forecast
fig, ax = plt.subplots()
ax.plot(df['sales'].values, label='Actual')
ax.plot(range(len(df), len(df)+10), predictions, label='Forecast')
ax.legend()
st.pyplot(fig)

# Segmentation
st.subheader("Customer Segmentation")
segments = segment_customers(df)
st.write(segments.head())

# Plot segments
fig2, ax2 = plt.subplots()
ax2.scatter(segments['customer_id'], segments['sales'], c=segments['segment'])
st.pyplot(fig2)

# ------------------------------
# AI INSIGHT (RULE BASED)
# ------------------------------
st.subheader("AI Insights")

avg_sales = df['sales'].mean()
if avg_sales < 250:
    st.write("⚠️ Sales are below average. Consider promotional campaigns.")
else:
    st.write("✅ Sales are performing well.")

# ==============================
# END
# ==============================
