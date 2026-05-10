import streamlit as st

def inject_cyber_css():
    st.markdown("""
        <style>
        @import url('https://rsms.me/inter/inter.css');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #050505; color: #E0E0E0; }
        
        /* Card de KPI estilo High-End */
        .kpi-card {
            background: rgba(0, 209, 255, 0.05);
            border: 1px solid rgba(0, 209, 255, 0.2);
            border-left: 5px solid #00D1FF;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Navegación lateral limpia */
        [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #1A1A1A; }
        </style>
    """, unsafe_allow_html=True)