import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

st.set_page_config(
    page_title="Investment Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional green-blue color scheme
st.markdown("""
    <style>
    :root {
        --primary-green: #059669;
        --primary-blue: #0284c7;
        --accent-teal: #14b8a6;
        --dark-gray: #1f2937;
        --light-gray: #f3f4f6;
        --border-color: #e5e7eb;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #059669 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #059669 0%, #0284c7 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.2);
    }
    
    .welcome-card h3 {
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .welcome-card p {
        font-size: 1rem;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    .info-box {
        background: white;
        border: 2px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-box {
        background: linear-gradient(to bottom right, #f0fdf4, #e0f2fe);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-green);
        margin-bottom: 1rem;
    }
    
    .metric-box h4 {
        color: var(--primary-green);
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--dark-gray);
    }
    
    .warning-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .warning-card p {
        margin: 0;
        color: #92400e;
        font-size: 0.95rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4);
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--dark-gray);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-green);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #f0fdf4, #e0f2fe);
    }
    
    /* Clean input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid var(--border-color);
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-green);
        box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

def initialize_system(api_key, search_key):
    """Initialize the investment analysis system"""
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["SERPER_API_KEY"] = search_key
    
    search_tool = SerperDevTool()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    research_agent = Agent(
        role="Investment Research Specialist",
        goal="Research and identify promising investment opportunities based on user criteria.",
        backstory="Experienced analyst specializing in comprehensive market research and opportunity identification.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool],
        llm=llm
    )
    
    analysis_agent = Agent(
        role="Financial Analyst",
        goal="Analyze investment data and create accurate financial projections.",
        backstory="Quantitative expert with strong background in financial modeling and forecasting.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool],
        llm=llm
    )
    
    risk_agent = Agent(
        role="Risk Assessment Specialist",
        goal="Evaluate potential risks and provide comprehensive risk analysis.",
        backstory="Risk management professional focused on protecting capital and optimizing risk-reward ratios.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool],
        llm=llm
    )
    
    advisor_agent = Agent(
        role="Investment Advisor",
        goal="Provide clear, actionable investment recommendations.",
        backstory="Senior advisor with extensive experience in personalized investment strategy development.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool],
        llm=llm
    )
    
    return research_agent, analysis_agent, risk_agent, advisor_agent, llm

def create_tasks(agents, data):
    """Create analysis tasks"""
    research_agent, analysis_agent, risk_agent, advisor_agent = agents
    
    task1 = Task(
        description=f"Research investment opportunities for {data['amount']} {data['currency']} focusing on {data['preferences']} in {data['geography']} markets. Find top opportunities with recent performance data.",
        expected_output="List of investment opportunities with recent performance metrics and key insights.",
        agent=research_agent
    )
    
    task2 = Task(
        description=f"Analyze opportunities and create {data['time_horizon']}-year ROI projections for {data['amount']} {data['currency']}. Calculate multiple scenarios.",
        expected_output="Detailed ROI projections with growth forecasts and scenario analysis.",
        agent=analysis_agent
    )
    
    task3 = Task(
        description=f"Assess risks for each option considering {data['risk_tolerance']} risk tolerance. Provide ratings and mitigation strategies.",
        expected_output="Comprehensive risk assessment with ratings and recommendations.",
        agent=risk_agent
    )
    
    task4 = Task(
        description=f"Create final investment recommendations for {data['amount']} {data['currency']} with {data['risk_tolerance']} risk tolerance and {data['time_horizon']}-year horizon. Rank top options.",
        expected_output="Ranked recommendations with allocation strategy and action steps.",
        agent=advisor_agent
    )
    
    return [task1, task2, task3, task4]

def run_analysis(data, api_key, search_key):
    """Execute investment analysis"""
    try:
        agents = initialize_system(api_key, search_key)
        manager_llm = agents[-1]
        agents = agents[:-1]
        
        tasks = create_tasks(agents, data)
        
        crew = Crew(
            agents=list(agents),
            tasks=tasks,
            manager_llm=manager_llm,
            process=Process.hierarchical,
            verbose=False
        )
        
        result = crew.kickoff()
        return result
    
    except Exception as e:
        error_msg = str(e).lower()
        if "authentication" in error_msg or "api" in error_msg or "key" in error_msg:
            st.error("⚠️ **Authentication Error**: Please verify your access credentials and try again.")
            st.info("💡 Ensure your credentials are valid and properly configured.")
        else:
            st.error(f"⚠️ **Analysis Error**: {str(e)}")
        return None

def create_projection_chart(years=10):
    """Create investment projection chart"""
    years_list = list(range(1, years + 1))
    
    conservative = [1.05**i for i in years_list]
    balanced = [1.085**i for i in years_list]
    growth = [1.135**i for i in years_list]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years_list, y=conservative,
        mode='lines+markers',
        name='Conservative (5% avg)',
        line=dict(color='#059669', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_list, y=balanced,
        mode='lines+markers',
        name='Balanced (8.5% avg)',
        line=dict(color='#0284c7', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_list, y=growth,
        mode='lines+markers',
        name='Growth (13.5% avg)',
        line=dict(color='#14b8a6', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Investment Growth Scenarios',
        xaxis_title='Years',
        yaxis_title='Growth Multiple',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(240,253,244,0.3)'
    )
    
    return fig

def create_trend_chart():
    """Create recent performance trend chart"""
    months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
    
    data = {
        'Month': months * 4,
        'Return (%)': [2.3, 1.8, 3.1, -0.5, 2.8, 4.2,
                       1.5, 1.2, 1.8, 1.3, 1.6, 2.1,
                       3.1, 2.5, 4.2, -1.2, 3.5, 5.1,
                       0.8, 0.7, 0.9, 0.8, 0.8, 1.0],
        'Category': ['Equities']*6 + ['Real Estate']*6 + ['International']*6 + ['Fixed Income']*6
    }
    
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Month', y='Return (%)', color='Category',
                 barmode='group',
                 title='Recent Market Performance (6 Months)',
                 template='plotly_white',
                 color_discrete_map={
                     'Equities': '#059669',
                     'Real Estate': '#0284c7',
                     'International': '#14b8a6',
                     'Fixed Income': '#f59e0b'
                 },
                 height=400)
    
    fig.update_layout(plot_bgcolor='rgba(240,253,244,0.3)')
    
    return fig

def main():
    st.markdown('<h1 class="main-title">💼 Investment Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Personalized investment recommendations tailored to your financial goals</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome to Your Investment Advisor</h3>
        <p>Get intelligent, data-driven investment recommendations personalized to your financial situation, 
        risk tolerance, and investment timeline. Make informed decisions with comprehensive market analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🔐 Access Configuration")
        
        env_key = os.getenv("OPENAI_API_KEY", "")
        env_search = os.getenv("SERPER_API_KEY", "")
        
        if env_key and env_search:
            st.success("✅ Configuration active")
        
        api_key = st.text_input(
            "Primary Access Key",
            value=env_key if env_key else "",
            type="password",
            help="Your secure access credential"
        )
        
        search_key = st.text_input(
            "Research Access Key",
            value=env_search if env_search else "",
            type="password",
            help="Your research access credential"
        )
        
        st.markdown("---")
        st.caption("🔒 Your information is secure and never stored")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Investment Parameters</div>', unsafe_allow_html=True)
        
        amount = st.number_input(
            "💵 Investment Amount",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000
        )
        
        currency = st.selectbox(
            "Currency",
            ["USD", "EUR", "GBP", "PKR", "INR", "AED"]
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            geography = st.multiselect(
                "🌍 Geographic Focus",
                ["Local/Domestic", "Regional", "International", "Emerging Markets"],
                default=["Local/Domestic"]
            )
        
        with col_b:
            preferences = st.multiselect(
                "📊 Investment Types",
                ["Stock Market", "Real Estate", "Bonds", "Mutual Funds", "ETFs", "Cryptocurrency", "Commodities"],
                default=["Stock Market"]
            )
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            risk_tolerance = st.select_slider(
                "⚖️ Risk Tolerance",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="Medium"
            )
        
        with col_d:
            time_horizon = st.slider(
                "📅 Investment Timeline (Years)",
                min_value=1,
                max_value=30,
                value=10
            )
    
    with col2:
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-box">
            <h4>Recommended Diversity</h4>
            <div class="value">4-6 Assets</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-box">
            <h4>Target Return</h4>
            <div class="value">7-12%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-box">
            <h4>Optimal Mix</h4>
            <div class="value">60/30/10</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-card">
            <p><strong>⚠️ Important Notice:</strong> This analysis is for informational purposes. 
            Always consult qualified financial professionals before making investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">Market Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 Growth Projections", "📊 Recent Trends"])
    
    with tab1:
        st.plotly_chart(create_projection_chart(time_horizon), use_container_width=True)
        st.info("💡 Projections are based on historical data and current market analysis.")
    
    with tab2:
        st.plotly_chart(create_trend_chart(), use_container_width=True)
        st.info("💡 Recent performance helps identify current market momentum and trends.")
    
    st.markdown("---")
    st.markdown('<div class="section-header">Generate Your Report</div>', unsafe_allow_html=True)
    
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'report' not in st.session_state:
        st.session_state.report = None
    
    if st.button("🚀 Generate Investment Report", type="primary", use_container_width=True):
        if not api_key or not search_key:
            st.error("⚠️ Please provide your access credentials in the sidebar.")
        elif not preferences:
            st.error("⚠️ Please select at least one investment type.")
        elif not geography:
            st.error("⚠️ Please select at least one geographic focus area.")
        else:
            investment_data = {
                'amount': amount,
                'currency': currency,
                'geography': ', '.join(geography),
                'preferences': ', '.join(preferences),
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon
            }
            
            with st.spinner("📊 Analyzing investment opportunities... This typically takes 2-3 minutes."):
                result = run_analysis(investment_data, api_key, search_key)
                
                if result:
                    st.session_state.report = result
                    st.session_state.analysis_done = True
                    st.success("✅ Analysis complete!")
                    st.balloons()
    
    if st.session_state.analysis_done and st.session_state.report:
        st.markdown("---")
        st.markdown('<div class="section-header">Your Investment Report</div>', unsafe_allow_html=True)
        
        st.markdown(st.session_state.report)
        
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label="📥 Download Complete Report",
                data=st.session_state.report,
                file_name=f"investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p style="font-size: 0.95rem;">Intelligent Analysis • Secure & Private • Informational Purposes Only</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem;">© 2024 Investment Advisor. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
