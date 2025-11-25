import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Investment Advisor (Claude-Powered)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .claude-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

def initialize_tools_and_agents(claude_api_key, serper_key, model_name="claude-sonnet-4-20250514"):
    """Initialize CrewAI agents with Claude models"""
    os.environ["ANTHROPIC_API_KEY"] = claude_api_key
    os.environ["SERPER_API_KEY"] = serper_key
    
    search_tool = SerperDevTool()
    
    # Initialize Claude LLM
    claude_llm = ChatAnthropic(
        model=model_name,
        temperature=0.7,
        anthropic_api_key=claude_api_key
    )
    
    # Market Research Agent
    market_research_agent = Agent(
        role="Market Research Analyst",
        goal="Research and analyze investment opportunities across different markets "
             "including stocks, real estate, and international markets based on user preferences.",
        backstory="Expert in global financial markets with deep knowledge of both "
                  "traditional and emerging investment opportunities. Specializes in "
                  "identifying high-potential investments based on current market trends.",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        llm=claude_llm
    )
    
    # Data Analysis Agent
    data_analyst_agent = Agent(
        role="Investment Data Analyst",
        goal="Analyze historical performance data, market trends, and generate "
             "ROI projections for recommended investment options.",
        backstory="Quantitative analyst with expertise in statistical modeling and "
                  "predictive analytics. Specializes in creating accurate financial "
                  "projections and risk assessments.",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        llm=claude_llm
    )
    
    # Risk Assessment Agent
    risk_analyst_agent = Agent(
        role="Risk Assessment Specialist",
        goal="Evaluate risks associated with different investment options and "
             "provide comprehensive risk analysis.",
        backstory="Experienced risk manager with deep understanding of market volatility, "
                  "geopolitical factors, and regulatory environments affecting investments.",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        llm=claude_llm
    )
    
    # Investment Advisor Agent
    advisor_agent = Agent(
        role="Senior Investment Advisor",
        goal="Synthesize all research and analysis to provide personalized investment "
             "recommendations with clear rationale and expected returns.",
        backstory="Senior financial advisor with 15+ years of experience in wealth "
                  "management and portfolio optimization. Known for making sound "
                  "investment recommendations tailored to client needs.",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        llm=claude_llm
    )
    
    return market_research_agent, data_analyst_agent, risk_analyst_agent, advisor_agent, search_tool, claude_llm

def create_tasks(agents, investment_data):
    """Create tasks for the crew"""
    market_research_agent, data_analyst_agent, risk_analyst_agent, advisor_agent = agents
    
    # Task 1: Market Research
    research_task = Task(
        description=(
            f"Research current investment opportunities for an investment amount of "
            f"{investment_data['amount']} {investment_data['currency']}. "
            f"Focus on: {investment_data['preferences']}. "
            f"Geographic preference: {investment_data['geography']}. "
            f"Find the top 5-7 investment options in each category. "
            f"Include recent 6-month performance data and current market conditions. "
            f"Identify top companies or opportunities in each sector."
        ),
        expected_output=(
            "A comprehensive report listing top investment opportunities with: "
            "1) Investment vehicle name/type "
            "2) Recent 6-month performance data "
            "3) Current market position "
            "4) Key companies or funds to consider "
            "5) Minimum investment requirements"
        ),
        agent=market_research_agent
    )
    
    # Task 2: Data Analysis and ROI Projection
    analysis_task = Task(
        description=(
            "Analyze the investment opportunities identified and create ROI projections "
            f"for 1-10 years for the investment amount of {investment_data['amount']} {investment_data['currency']}. "
            "Use historical data and current market trends. "
            "Calculate best-case, average-case, and worst-case scenarios. "
            "Include factors like compound growth, dividends, and market volatility."
        ),
        expected_output=(
            "Detailed ROI projections with: "
            "1) Year-by-year projected returns for 1-10 years "
            "2) Best, average, and worst-case scenarios "
            "3) Expected annual growth rates "
            "4) Total projected value at each year milestone "
            "5) Key assumptions used in projections"
        ),
        agent=data_analyst_agent
    )
    
    # Task 3: Risk Assessment
    risk_task = Task(
        description=(
            "Evaluate risks for each recommended investment option. "
            f"Consider the user's risk tolerance: {investment_data['risk_tolerance']}. "
            "Analyze market volatility, liquidity, regulatory risks, and external factors. "
            "Provide risk ratings (Low/Medium/High) for each option."
        ),
        expected_output=(
            "Risk assessment report including: "
            "1) Risk rating for each investment option "
            "2) Key risk factors identified "
            "3) Risk mitigation strategies "
            "4) Suitability based on user's risk tolerance "
            "5) Diversification recommendations"
        ),
        agent=risk_analyst_agent
    )
    
    # Task 4: Final Recommendation
    recommendation_task = Task(
        description=(
            "Synthesize all research, analysis, and risk assessments to provide "
            "final investment recommendations. Rank the top 3-5 options. "
            "Provide clear rationale for each recommendation. "
            f"Consider investment amount: {investment_data['amount']} {investment_data['currency']}, "
            f"risk tolerance: {investment_data['risk_tolerance']}, "
            f"and time horizon: {investment_data['time_horizon']} years."
        ),
        expected_output=(
            "Final investment recommendation report with: "
            "1) Top 3-5 ranked investment recommendations "
            "2) Allocation strategy (% of total investment) "
            "3) Expected returns for each recommendation "
            "4) Clear rationale and supporting data "
            "5) Action steps to get started "
            "6) Summary table comparing all options"
        ),
        agent=advisor_agent
    )
    
    return [research_task, analysis_task, risk_task, recommendation_task]

def run_investment_analysis(investment_data, claude_api_key, serper_key, model_name):
    """Run the complete investment analysis"""
    try:
        # Initialize agents
        agents = initialize_tools_and_agents(claude_api_key, serper_key, model_name)
        claude_llm = agents[-1]
        search_tool = agents[-2]
        agents = agents[:-2]
        
        # Create tasks
        tasks = create_tasks(agents, investment_data)
        
        # Create crew
        investment_crew = Crew(
            agents=list(agents),
            tasks=tasks,
            manager_llm=claude_llm,
            process=Process.hierarchical,
            verbose=True
        )
        
        # Execute analysis
        result = investment_crew.kickoff()
        return result
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        return None

def create_roi_visualization(years=10):
    """Create sample ROI visualization"""
    years_list = list(range(1, years + 1))
    
    # Sample data for different investment types
    stock_market_roi = [1.08**i for i in years_list]  # 8% annual return
    real_estate_roi = [1.06**i for i in years_list]  # 6% annual return
    international_roi = [1.10**i for i in years_list]  # 10% annual return
    bonds_roi = [1.04**i for i in years_list]  # 4% annual return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years_list, y=stock_market_roi,
        mode='lines+markers',
        name='Stock Market (8% avg)',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_list, y=real_estate_roi,
        mode='lines+markers',
        name='Real Estate (6% avg)',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_list, y=international_roi,
        mode='lines+markers',
        name='International (10% avg)',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_list, y=bonds_roi,
        mode='lines+markers',
        name='Bonds (4% avg)',
        line=dict(color='#d62728', width=3)
    ))
    
    fig.update_layout(
        title='Projected ROI Growth (Multiplier) - Next 10 Years',
        xaxis_title='Years',
        yaxis_title='Return Multiplier',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_six_month_performance():
    """Create 6-month performance visualization"""
    months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
    
    data = {
        'Month': months * 4,
        'Return (%)': [2.3, 1.8, 3.1, -0.5, 2.8, 4.2,  # Stock Market
                       1.5, 1.2, 1.8, 1.3, 1.6, 2.1,   # Real Estate
                       3.1, 2.5, 4.2, -1.2, 3.5, 5.1,  # International
                       0.8, 0.7, 0.9, 0.8, 0.8, 1.0],  # Bonds
        'Category': ['Stock Market']*6 + ['Real Estate']*6 + ['International']*6 + ['Bonds']*6
    }
    
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Month', y='Return (%)', color='Category',
                 barmode='group',
                 title='Last 6 Months Performance by Investment Category',
                 template='plotly_white',
                 height=400)
    
    return fig

def main():
    st.markdown('<h1 class="main-header">💰 AI-Powered Investment Advisor</h1>', unsafe_allow_html=True)
    
    st.markdown('<p style="text-align: center;"><span class="claude-badge">⚡ Powered by Claude Sonnet 4</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This intelligent investment advisor uses Claude AI agents to analyze global investment 
    opportunities and provide personalized recommendations based on your preferences and goals.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 API Configuration")
        st.markdown("Get your API keys:")
        st.markdown("- [Anthropic API Key](https://console.anthropic.com/)")
        st.markdown("- [Serper API Key](https://serper.dev/)")
        
        # Check for environment variables first
        env_claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        env_serper_key = os.getenv("SERPER_API_KEY", "")
        
        # Show info if keys are loaded from .env
        if env_claude_key and env_serper_key:
            st.success("✅ API keys loaded from .env file")
            st.info("You can override them below if needed")
        
        claude_api_key = st.text_input(
            "Anthropic API Key", 
            value=env_claude_key if env_claude_key else "",
            type="password",
            help="Enter your Anthropic API key or set ANTHROPIC_API_KEY in .env file"
        )
        serper_key = st.text_input(
            "Serper API Key", 
            value=env_serper_key if env_serper_key else "",
            type="password",
            help="Enter your Serper API key or set SERPER_API_KEY in .env file"
        )
        
        st.markdown("---")
        st.markdown("### Claude Model Selection")
        model_name = st.selectbox(
            "Choose Claude Model",
            [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-sonnet-3-5-20241022",
                "claude-3-5-sonnet-20241022"
            ],
            help="Claude Sonnet 4 is recommended for best quality"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This app uses Claude AI agents to research and analyze investment opportunities across different markets.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Investment Details</h2>', unsafe_allow_html=True)
        
        amount = st.number_input(
            "Investment Amount",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
            help="Enter the amount you want to invest"
        )
        
        currency = st.selectbox(
            "Currency",
            ["USD", "EUR", "GBP", "PKR", "INR", "AED"],
            help="Select your investment currency"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            geography = st.multiselect(
                "Geographic Preference",
                ["Local/Domestic", "Regional", "International", "Emerging Markets"],
                default=["Local/Domestic"],
                help="Select your geographic investment preference"
            )
        
        with col_b:
            preferences = st.multiselect(
                "Investment Preferences",
                ["Stock Market", "Real Estate", "Bonds", "Mutual Funds", "ETFs", "Cryptocurrency", "Commodities"],
                default=["Stock Market"],
                help="Select your preferred investment types"
            )
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="Medium",
                help="Your comfort level with investment risk"
            )
        
        with col_d:
            time_horizon = st.slider(
                "Investment Time Horizon (Years)",
                min_value=1,
                max_value=30,
                value=10,
                help="How long do you plan to hold this investment?"
            )
    
    with col2:
        st.markdown('<h2 class="sub-header">Quick Stats</h2>', unsafe_allow_html=True)
        
        # Sample metrics
        st.metric("Recommended Diversification", "4-6 Assets")
        st.metric("Average Expected Return", "7-12% annually")
        st.metric("Optimal Portfolio Balance", "60/30/10")
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Disclaimer:</strong> This is an AI-generated analysis. 
        Always consult with a certified financial advisor before making investment decisions.
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations section
    st.markdown('<h2 class="sub-header">📊 Historical Performance & Projections</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 ROI Projections", "📉 Recent Performance"])
    
    with tab1:
        st.plotly_chart(create_roi_visualization(time_horizon), use_container_width=True)
        
        st.markdown("""
        **Understanding ROI Projections:**
        - These projections show potential growth multipliers over your selected time horizon
        - Based on historical average returns for each investment category
        - Actual returns may vary based on market conditions
        """)
    
    with tab2:
        st.plotly_chart(create_six_month_performance(), use_container_width=True)
        
        st.markdown("""
        **Recent Market Trends:**
        - Last 6 months performance across major investment categories
        - Helps understand current market momentum
        - Consider this alongside long-term projections
        """)
    
    # Analysis button
    st.markdown('<h2 class="sub-header">🎯 Get Your Personalized Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Start Claude AI Analysis", type="primary", use_container_width=True):
        if not claude_api_key or not serper_key:
            st.error("⚠️ Please provide both Anthropic and Serper API keys in the sidebar.")
        elif not preferences:
            st.error("⚠️ Please select at least one investment preference.")
        elif not geography:
            st.error("⚠️ Please select at least one geographic preference.")
        else:
            investment_data = {
                'amount': amount,
                'currency': currency,
                'geography': ', '.join(geography),
                'preferences': ', '.join(preferences),
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon
            }
            
            with st.spinner("🤖 Claude AI agents are analyzing investment opportunities... This may take 2-3 minutes."):
                result = run_investment_analysis(investment_data, claude_api_key, serper_key, model_name)
                
                if result:
                    st.session_state.results = result
                    st.session_state.analysis_complete = True
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.results:
        st.markdown('<h2 class="sub-header">📋 Your Personalized Investment Report</h2>', unsafe_allow_html=True)
        
        st.markdown(st.session_state.results)
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=st.session_state.results,
            file_name=f"investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Powered by Claude AI Multi-Agent System</strong></p>
    <p>Using CrewAI, Anthropic Claude, and Serper for comprehensive investment analysis</p>
    <p style="font-size: 0.8rem;">⚠️ This tool provides information only. Not financial advice. Invest responsibly.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
