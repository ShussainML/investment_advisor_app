"""
Helper utilities for the Investment Advisor App
"""

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def calculate_compound_growth(principal, rate, years):
    """Calculate compound growth over years"""
    return principal * (1 + rate) ** years

def generate_roi_scenarios(amount, years=10):
    """Generate best, average, and worst case ROI scenarios"""
    scenarios = {
        'years': list(range(1, years + 1)),
        'conservative': [],
        'moderate': [],
        'aggressive': []
    }
    
    # Conservative: 4-6% annual return
    conservative_rate = 0.05
    # Moderate: 7-10% annual return
    moderate_rate = 0.085
    # Aggressive: 12-15% annual return
    aggressive_rate = 0.135
    
    for year in scenarios['years']:
        scenarios['conservative'].append(
            calculate_compound_growth(amount, conservative_rate, year)
        )
        scenarios['moderate'].append(
            calculate_compound_growth(amount, moderate_rate, year)
        )
        scenarios['aggressive'].append(
            calculate_compound_growth(amount, aggressive_rate, year)
        )
    
    return scenarios

def create_detailed_roi_chart(amount, years, currency):
    """Create detailed ROI projection chart"""
    scenarios = generate_roi_scenarios(amount, years)
    
    fig = go.Figure()
    
    # Conservative scenario
    fig.add_trace(go.Scatter(
        x=scenarios['years'],
        y=scenarios['conservative'],
        mode='lines+markers',
        name='Conservative (5% avg)',
        line=dict(color='#2ecc71', width=3),
        hovertemplate=f'Year %{{x}}<br>{currency} %{{y:,.0f}}<extra></extra>'
    ))
    
    # Moderate scenario
    fig.add_trace(go.Scatter(
        x=scenarios['years'],
        y=scenarios['moderate'],
        mode='lines+markers',
        name='Moderate (8.5% avg)',
        line=dict(color='#3498db', width=3),
        hovertemplate=f'Year %{{x}}<br>{currency} %{{y:,.0f}}<extra></extra>'
    ))
    
    # Aggressive scenario
    fig.add_trace(go.Scatter(
        x=scenarios['years'],
        y=scenarios['aggressive'],
        mode='lines+markers',
        name='Aggressive (13.5% avg)',
        line=dict(color='#e74c3c', width=3),
        hovertemplate=f'Year %{{x}}<br>{currency} %{{y:,.0f}}<extra></extra>'
    ))
    
    # Add initial investment line
    fig.add_hline(
        y=amount,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial: {currency} {amount:,.0f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f'Investment Growth Projection ({years} Years)',
        xaxis_title='Years',
        yaxis_title=f'Portfolio Value ({currency})',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def generate_sector_allocation():
    """Generate recommended sector allocation"""
    sectors = ['Technology', 'Healthcare', 'Finance', 'Real Estate', 'Consumer', 'Energy', 'Other']
    conservative = [15, 20, 25, 15, 15, 5, 5]
    moderate = [25, 15, 20, 15, 15, 5, 5]
    aggressive = [35, 15, 15, 10, 15, 5, 5]
    
    return {
        'sectors': sectors,
        'conservative': conservative,
        'moderate': moderate,
        'aggressive': aggressive
    }

def create_allocation_chart(risk_tolerance):
    """Create sector allocation pie chart"""
    allocation = generate_sector_allocation()
    
    if risk_tolerance in ['Very Low', 'Low']:
        values = allocation['conservative']
        title = 'Conservative Portfolio Allocation'
    elif risk_tolerance in ['High', 'Very High']:
        values = allocation['aggressive']
        title = 'Aggressive Portfolio Allocation'
    else:
        values = allocation['moderate']
        title = 'Moderate Portfolio Allocation'
    
    fig = go.Figure(data=[go.Pie(
        labels=allocation['sectors'],
        values=values,
        hole=0.3,
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True,
        height=400
    )
    
    return fig

def generate_monthly_performance(months=6):
    """Generate realistic monthly performance data"""
    categories = ['Stocks', 'Bonds', 'Real Estate', 'Commodities']
    
    data = []
    for i in range(months):
        month = (datetime.now() - timedelta(days=30*(months-i))).strftime('%b %Y')
        for category in categories:
            # Generate realistic returns with some volatility
            if category == 'Stocks':
                base_return = np.random.normal(1.2, 2.5)
            elif category == 'Bonds':
                base_return = np.random.normal(0.5, 0.8)
            elif category == 'Real Estate':
                base_return = np.random.normal(0.8, 1.2)
            else:  # Commodities
                base_return = np.random.normal(0.3, 3.0)
            
            data.append({
                'Month': month,
                'Category': category,
                'Return (%)': round(base_return, 2)
            })
    
    return pd.DataFrame(data)

def format_currency(amount, currency):
    """Format currency with appropriate symbols"""
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'PKR': 'Rs.',
        'INR': '₹',
        'AED': 'AED '
    }
    
    symbol = symbols.get(currency, currency + ' ')
    
    if amount >= 1_000_000:
        return f"{symbol}{amount/1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"{symbol}{amount/1_000:.2f}K"
    else:
        return f"{symbol}{amount:.2f}"

def calculate_diversification_score(preferences):
    """Calculate diversification score based on selected preferences"""
    max_score = 7  # Total number of investment types
    current = len(preferences)
    
    if current >= 5:
        return "Excellent", "🌟"
    elif current >= 3:
        return "Good", "✅"
    elif current >= 2:
        return "Moderate", "⚠️"
    else:
        return "Low", "❌"

def get_risk_tooltip(risk_level):
    """Get detailed tooltip for risk levels"""
    tooltips = {
        'Very Low': "Suitable for capital preservation. Expects 3-5% annual returns with minimal volatility.",
        'Low': "Conservative approach. Expects 4-7% annual returns with low volatility.",
        'Medium': "Balanced risk-reward. Expects 7-10% annual returns with moderate volatility.",
        'High': "Growth-focused. Expects 10-15% annual returns with higher volatility.",
        'Very High': "Aggressive strategy. Expects 15%+ annual returns with significant volatility."
    }
    return tooltips.get(risk_level, "")

def generate_investment_summary(investment_data):
    """Generate a summary of investment parameters"""
    summary = f"""
    ### 📊 Investment Summary
    
    **Investment Amount**: {format_currency(investment_data['amount'], investment_data['currency'])}
    
    **Investment Preferences**: {investment_data['preferences']}
    
    **Geographic Focus**: {investment_data['geography']}
    
    **Risk Profile**: {investment_data['risk_tolerance']}
    
    **Time Horizon**: {investment_data['time_horizon']} years
    
    **Expected Return Range**: {get_expected_return_range(investment_data['risk_tolerance'])}
    """
    return summary

def get_expected_return_range(risk_tolerance):
    """Get expected return range based on risk tolerance"""
    ranges = {
        'Very Low': "3-5% annually",
        'Low': "4-7% annually",
        'Medium': "7-10% annually",
        'High': "10-15% annually",
        'Very High': "15%+ annually"
    }
    return ranges.get(risk_tolerance, "7-10% annually")

def validate_investment_inputs(amount, preferences, geography):
    """Validate user inputs"""
    errors = []
    
    if amount < 1000:
        errors.append("Investment amount should be at least 1,000")
    
    if amount > 10_000_000:
        errors.append("Investment amount exceeds maximum limit of 10,000,000")
    
    if not preferences:
        errors.append("Please select at least one investment preference")
    
    if not geography:
        errors.append("Please select at least one geographic preference")
    
    return errors

def create_comparison_table(recommendations):
    """Create a comparison table for investment recommendations"""
    # This would process the AI recommendations into a structured table
    # For now, return a sample structure
    data = {
        'Investment Type': [],
        'Expected Return': [],
        'Risk Level': [],
        'Time to Maturity': [],
        'Liquidity': []
    }
    return pd.DataFrame(data)
