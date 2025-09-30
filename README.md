# Content-Monetization-Modeler
Content Monetization Modeler - Machine Learning

**Problem Statement:**

This project focuses on predicting YouTube Ad Revenue using **machine learning models**.  

The final solution includes:

- A trained linear regression model
- 
- A **Streamlit Web App** for predictions
- 
- Data preprocessing and feature engineering pipeline
- 
- Documentation & insights

**Technology Stack Used:**

Python

Streamlit

Scikit-learn

Google API Client (YouTube Data API)

Matplotlib & Seaborn (EDA & Visualization)

Pandas, NumPy, SciPy (Data Handling)

Joblib (Model Persistence)

Visual Studio Code

**Installation:**

pip install pandas

pip install numpy

pip install scikit-learn

pip install streamlit

pip install matplotlib

pip install seaborn

pip install google-api-python-client

pip install joblib

pip install isodate

pip install scipy

**Import Libraries:**
# Core Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import joblib

import os

# Streamlit & API

import streamlit as st

from googleapiclient.discovery import build

import isodate

# Sklearn Models & Tools

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Preprocessing & Pipelines

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

# Statistics

from scipy import stats

**Approach:**

**1. Data Extraction & Understanding**

Dataset: YouTube Monetization Modeler (~122,000 rows, CSV)

Features: Views, Likes, Comments, Watch Time, Subscribers, Category, Device, Country, etc.

Target: Ad Revenue (USD)

**2. Data Cleaning & Preprocessing**

Handled ~5% missing values

Removed ~2% duplicates

Encoded categorical variables (category, device, country)

Normalized/Scaled features where needed

**3. Feature Engineering**

engagement rate = (likes+comments)/views

avg_watch_time_per_view = watch_time_minutes/ views

**4. Exploratory Data Analysis (EDA)**

Strong correlation: likes, watch time → revenue

Engagement rate proved critical for revenue prediction

**5. Model Building**

Trained & compared 5 regression models:

LinearRegression

DecisionTreeRegressor

RandomForestRegressor

GradientBoostingRegressor

XGBRegressor

Best model: Linear Regression

Performance:
R² ≈ 0.953
RMSE ≈ 13.480
MAE ≈ 3.111

**6. Deployment (Streamlit App)**

Two Modes of Prediction:

Manual Input

YouTube Link (via YouTube API fetch)

Predicts Ad Revenue in USD instantly

Simple, interactive UI with metrics display

**Snapshot:**

**Manual Input:**

<img width="1918" height="1008" alt="image" src="https://github.com/user-attachments/assets/cc05e1f1-b5a6-4e27-b39d-0c4c32c9abb7" />

<img width="1918" height="1011" alt="image" src="https://github.com/user-attachments/assets/e54157d9-133a-4111-a2f3-fa71d08c6a00" />

**Youtube Link:**

<img width="1908" height="996" alt="image" src="https://github.com/user-attachments/assets/aea55c60-1467-4c02-aa14-6d0983fb1036" />





