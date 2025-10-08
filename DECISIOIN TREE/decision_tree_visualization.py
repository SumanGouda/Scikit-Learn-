import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import streamlit as st

st.title("Regression Tree Visualizer")

def file_upload_with_preview():
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show success message
            st.success(f"✅ Successfully loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Show data preview in tabs
            tab1, tab2 = st.tabs(["Data Head", "Data Info"])
            
            with tab1:
                st.dataframe(df.head(10))
            
            with tab2:
                st.write(f"**Shape:** {df.shape}")
                st.write("**Columns:**", list(df.columns))
            
            return df
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    
    return None

df = file_upload_with_preview()

if df is not None:
    with st.expander("📊 Next Steps", expanded=True):
        st.write("Please select 2 columns from the dropdown menu below")

    feature1 = st.selectbox("Select dependent feature", df.columns)
    feature2 = st.selectbox("Select independent feature", df.columns)
    
    try:
        # Try to convert to numeric - if it fails, columns are not numerical
        pd.to_numeric(df[feature1])
        pd.to_numeric(df[feature2])
        
        # If no error, both are numerical
        st.success("✅ Both features are numerical - ready to train!")
        
        error = {}
        for i in range(df[feature1].shape[0] - 1):
            avg_x = (df[feature1][i] + df[feature1][i + 1]) / 2
            above_rows = df[df[feature1] > avg_x]
            below_rows = df[df[feature1] <= avg_x]
            
            # Calculate SSE for each split separately
            if len(above_rows) > 0 and len(below_rows) > 0:
                sse_above = np.sum(np.square(above_rows[feature2] - np.mean(above_rows[feature2])))
                sse_below = np.sum(np.square(below_rows[feature2] - np.mean(below_rows[feature2])))
                total_sse = sse_above + sse_below
                error[avg_x] = total_sse
                
        # Find the minimum error
        min_error = min(error.values())
        
        # Plot the error
        plt.figure(figsize=(10, 6))
        plt.plot(list(error.keys()), list(error.values()), 'ro')
        plt.xlabel(feature1)
        plt.ylabel("SSE")
        st.pyplot(plt.gcf())  
                
        
        
    except ValueError:
        st.error("❌ Both features must be numerical columns")