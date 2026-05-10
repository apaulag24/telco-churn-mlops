import streamlit as st
from components.eda_charts import plot_churn_donut

def render_overview(df):
    st.markdown("<h2 id='overview'>📋 Overview & KPIs</h2>", unsafe_allow_html=True)
    
    # Grid de 6 KPIs
    row1 = st.columns(3)
    row2 = st.columns(3)
    
    with row1[0]: st.metric("Total Customers", f"{len(df):,}")
    with row1[1]: st.metric("Churn Rate", f"{(df['Churn']=='Yes').mean()*100:.1f}%")
    with row1[2]: st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} mo")
    
    with row2[0]: st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.1f}")
    with row2[1]: st.metric("Max Total Charges", f"${df['TotalCharges'].max():.1f}")
    with row2[2]: st.metric("Senior Citizens", f"{(df['SeniorCitizen']==1).sum()}")

    # Schema Table
    st.write("---")
    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        st.plotly_chart(plot_churn_donut(df), use_container_width=True)
    with col_right:
        st.dataframe(df.dtypes.to_frame(name='Type'), use_container_width=True)