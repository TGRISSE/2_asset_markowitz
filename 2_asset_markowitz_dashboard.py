import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

###------
# Custom Colors for Plots
###------
TEXT_PETROL = "#00414B"         # Labels and text
TEXT_FADINGPETROL = "#B4C8CD"     # Grid color
TEXT_PETROLLIGHTER = "#4B7C7D"    # Plot color 1 
DIAGRAM_OPT1_BLUE = "#0072CE"     # Plot color 3
DIAGRAM_OPT1_RED = "#FF6A14"      # Plot color 4
DIAGRAM_OPT1_BRIGHTBLUE = "#00C8FF"  # Plot color 5
DIAGRAM_OPT1_ORANGE = "#FFC800"   # Plot color 6

DEBUG_MODE = False

###------
# Language Translations
###------
TRANSLATIONS = {
    'de': {
        'title': "2-Asset Markowitz Szenarien",
        'input_parameters': "Eingabeparameter",
        'portfolio_parameters': "Portfolio-Parameter:",
        'portfolio_return': "Portfolio-Rendite",
        'portfolio_volatility': "Portfolio-Volatilität",
        'new_asset_parameters': "Parameter des neuen Assets:",
        'new_asset_volatility': "Volatilität des neuen Assets",
        'new_asset_return': "Rendite des neuen Assets",
        'correlation': "Korrelation",
        'break_even_analysis': "Break-Even-Analyse",
        'break_even_condition': "Break-Even-Bedingung für die Aufnahme des neuen Assets:",
        'where': "wobei:",
        'break_even_explanation': """
        - μₐ: Rendite des neuen Assets
        - Sₚ: Sharpe-Ratio des Portfolios
        - ρ: Korrelation
        - σₐ: Volatilität des neuen Assets
        """,
        'break_even_return': "Break-Even-Rendite des neuen Assets (basierend auf Portfolio Sharpe von {:.2f}): **{:.2%}**",
        'efficient_frontier': "Effiziente Grenze",
        'optimal_weights': "Optimale Gewichtung",
        'portfolio': "Portfolio",
        'new_asset': "Neues Asset",
        'smith_analysis': "Thomas Smith's Gewichtungsverhältnis-Analyse ([Paper](https://riskyfinance.com/wp-content/uploads/2019/07/The_Sharpe_Ratio_Ratio_ThomasSmith-FINAL.pdf))",
        'smith_equation': "Das optimale Gewichtungsverhältnis kann mit Smith's Gleichung bestimmt werden:",
        'smith_explanation': """
        - w₂/w₁: Verhältnis von neuem Asset-Gewicht zu Portfolio-Gewicht
        - σ₁/σ₂: Verhältnis von Portfolio-Volatilität zu neuer Asset-Volatilität
        - SSR: Verhältnis der Sharpe-Ratio des neuen Assets zur Portfolio-Sharpe-Ratio (SSR = SR₂/SR₁)
        - ρ: Korrelation zwischen den Assets
        """,
        'sharpe_ratios': "**Sharpe-Ratios:**",
        'new_asset_sharpe': "- Sharpe-Ratio neues Asset: {:.4f}",
        'portfolio_sharpe': "- Portfolio Sharpe-Ratio: {:.4f}",
        'ssr': "- SSR (Sharpe-Ratio-Verhältnis): {:.4f}",
        'weight_ratio_components': "**Gewichtungsverhältnis-Komponenten:**",
        'volatility_ratio': "- Volatilitätsverhältnis (σ₁/σ₂): {:.4f}",
        'numerator': "- Zähler (SSR - ρ): {:.4f}",
        'denominator': "- Nenner (1 - SSR·ρ): {:.4f}",
        'weight_ratio': "- **Gewichtungsverhältnis (w₂/w₁): {:.4f}**",
        'smith_optimal_weights': "**Smith's optimale Gewichtungen:**",
        'smith_portfolio': "- Portfolio (w₁): {:.2%}",
        'smith_new_asset': "- Neues Asset (w₂): {:.2%}",
        'plot_title': 'Effiziente Grenze (Neues Asset + Portfolio, Long-Only)',
        'plot_volatility': 'Portfolio-Volatilität',
        'plot_return': 'Portfolio-Rendite',
        'plot_efficient_frontier': 'Effiziente Grenze',
        'plot_new_asset': 'Neues Asset',
        'plot_portfolio': 'Portfolio',
        'plot_optimal_portfolio': 'Optimales Portfolio',
        'plot_parameters': 'Parameter:\nKorrelation    = {:.2f}\nPortfolio Rend = {:.2%}\nPortfolio Vol  = {:.2%}\nAsset Rend     = {:.2%}\nAsset Vol      = {:.2%}'
    },
    'en': {
        'title': "2-Asset Markowitz Scenarios",
        'input_parameters': "Input Parameters",
        'portfolio_parameters': "Portfolio Parameters:",
        'portfolio_return': "Portfolio Return",
        'portfolio_volatility': "Portfolio Volatility",
        'new_asset_parameters': "New Asset Parameters:",
        'new_asset_volatility': "New Asset Volatility",
        'new_asset_return': "New Asset Return",
        'correlation': "Correlation",
        'break_even_analysis': "Break-even Analysis",
        'break_even_condition': "Break-even condition for adding the new asset:",
        'where': "where:",
        'break_even_explanation': """
        - μₐ: New asset return
        - Sₚ: Portfolio Sharpe ratio
        - ρ: Correlation
        - σₐ: New asset volatility
        """,
        'break_even_return': "Break-even New Asset Return (based on portfolio Sharpe of {:.2f}): **{:.2%}**",
        'efficient_frontier': "Efficient Frontier",
        'optimal_weights': "Optimal Portfolio Weights",
        'portfolio': "Portfolio",
        'new_asset': "New Asset",
        'smith_analysis': "Thomas Smith's Weight Ratio Analysis ([paper](https://riskyfinance.com/wp-content/uploads/2019/07/The_Sharpe_Ratio_Ratio_ThomasSmith-FINAL.pdf))",
        'smith_equation': "The optimal weight ratio can be determined using Smith's equation:",
        'smith_explanation': """
        - w₂/w₁: Ratio of new asset weight to portfolio weight
        - σ₁/σ₂: Ratio of portfolio volatility to new asset volatility
        - SSR: Ratio of new asset Sharpe ratio to portfolio Sharpe ratio (SSR = SR₂/SR₁)
        - ρ: Correlation between assets
        """,
        'sharpe_ratios': "**Sharpe Ratios:**",
        'new_asset_sharpe': "- New Asset Sharpe: {:.4f}",
        'portfolio_sharpe': "- Portfolio Sharpe: {:.4f}",
        'ssr': "- SSR (Sharpe Ratio Ratio): {:.4f}",
        'weight_ratio_components': "**Weight Ratio Components:**",
        'volatility_ratio': "- Volatility ratio (σ₁/σ₂): {:.4f}",
        'numerator': "- Numerator (SSR - ρ): {:.4f}",
        'denominator': "- Denominator (1 - SSR·ρ): {:.4f}",
        'weight_ratio': "- **Weight ratio (w₂/w₁): {:.4f}**",
        'smith_optimal_weights': "**Smith's Optimal Weights:**",
        'smith_portfolio': "- Portfolio (w₁): {:.2%}",
        'smith_new_asset': "- New Asset (w₂): {:.2%}",
        'plot_title': 'Efficient Frontier (New Asset + Portfolio, Long-Only)',
        'plot_volatility': 'Portfolio Volatility',
        'plot_return': 'Portfolio Return',
        'plot_efficient_frontier': 'Efficient Frontier',
        'plot_new_asset': 'New Asset',
        'plot_portfolio': 'Portfolio',
        'plot_optimal_portfolio': 'Optimal Portfolio',
        'plot_parameters': 'Parameters:\nCorrelation    = {:.2f}\nPortfolio Ret  = {:.2%}\nPortfolio Vol  = {:.2%}\nAsset Ret      = {:.2%}\nAsset Vol      = {:.2%}'
    }
}

###------
# Helper Functions
###------
def get_text(key, lang='de'):
    """Get translated text for the given key."""
    return TRANSLATIONS[lang][key]

def build_portfolio_functions(new_return, pf_return, new_vol, pf_vol, rho):
    """Return functions to compute portfolio return and variance."""
    cov_new_pf = rho * new_vol * pf_vol

    def portfolio_return(w):
        return w[0] * new_return + w[1] * pf_return

    def portfolio_variance(w):
        return w[0]**2 * new_vol**2 + w[1]**2 * pf_vol**2 + 2 * w[0] * w[1] * cov_new_pf

    return portfolio_return, portfolio_variance

def compute_optimal_weights(new_return, pf_return, new_vol, pf_vol, rho):
    """Compute optimal weights using the efficient frontier approach."""
    port_return_fn, port_variance_fn = build_portfolio_functions(new_return, pf_return, new_vol, pf_vol, rho)
    
    def objective(w):
        port_vol = np.sqrt(port_variance_fn(w))
        return -port_return_fn(w) / port_vol if port_vol != 0 else 1e6

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = ((0, 1), (0, 1))
    w0 = [0.5, 0.5]
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    return optimal_weights, port_return_fn, port_variance_fn

def plot_efficient_frontier(new_return, pf_return, new_vol, pf_vol, rho, optimal_weights, portfolio_return, portfolio_variance, lang='de'):
    """Plot the efficient frontier with translations."""
    n_points = 100
    weights = np.linspace(0, 1, n_points)
    returns = [portfolio_return([w, 1 - w]) for w in weights]
    volatilities = [np.sqrt(portfolio_variance([w, 1 - w])) for w in weights]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(volatilities, returns, '-', color=DIAGRAM_OPT1_BLUE, label=get_text('plot_efficient_frontier', lang))
    ax.scatter(new_vol, new_return, color=DIAGRAM_OPT1_ORANGE, marker='D', s=150, label=get_text('plot_new_asset', lang))
    ax.scatter(pf_vol, pf_return, color=DIAGRAM_OPT1_BRIGHTBLUE, marker='D', s=150, label=get_text('plot_portfolio', lang))
    ax.scatter(np.sqrt(portfolio_variance(optimal_weights)), portfolio_return(optimal_weights),
               color=DIAGRAM_OPT1_RED, marker='*', s=200, label=get_text('plot_optimal_portfolio', lang))
    
    ax.set_xlabel(get_text('plot_volatility', lang), color=TEXT_PETROL)
    ax.set_ylabel(get_text('plot_return', lang), color=TEXT_PETROL)
    ax.set_title(get_text('plot_title', lang), color=TEXT_PETROL, pad=20)
    ax.legend(edgecolor=TEXT_PETROL, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
    ax.grid(True, color=TEXT_FADINGPETROL)
    
    textstr = get_text('plot_parameters', lang).format(rho, pf_return, pf_vol, new_return, new_vol)
    ax.text(1.05, 0.5, textstr, transform=ax.transAxes,
            verticalalignment='center')
    
    plt.subplots_adjust(right=0.75)
    return fig

###------
# Streamlit Dashboard
###------
def main():
    st.set_page_config(page_title="2-Asset Markowitz", layout="wide")
    
    # Add CSS styling
    st.markdown(
        """
        <style>
        div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
            background: none !important;
        }
        div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
            background-color: rgb(0, 65, 75) !important;
            box-shadow: rgb(0 65 75 / 20%) 0px 0px 0px 0.2rem !important;
        }
        div.stSlider > div[data-baseweb="slider"] > div > div[data-testid="stSliderTrack"] > div:first-child {
            background: rgb(0, 65, 75) !important;
        }
        div.stSlider > div[data-baseweb="slider"] > div > div[data-testid="stSliderTrack"] > div:nth-of-type(2) {
            background: #ddd !important;
        }
        div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
            color: rgb(0, 65, 75) !important;
            background: none !important;
        }
        div[data-testid="column"] > div.stButton > button {
            background-color: rgb(0, 65, 75);
            color: white;
            border: none;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
        }
        div[data-testid="column"] > div.stButton > button:hover {
            background-color: rgb(0, 85, 95);
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize session state for language
    if 'language' not in st.session_state:
        st.session_state.language = 'de'  # Default to German
    
    # Create a row with title and language switch
    title_col, lang_col = st.columns([6, 1])
    with title_col:
        st.title(get_text('title', st.session_state.language))
    with lang_col:
        st.write("")
        st.write("")
        if st.button("DE/EN"):
            st.session_state.language = 'en' if st.session_state.language == 'de' else 'de'
            st.rerun()

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Input parameters (left column)
    with col1:
        st.subheader(get_text('input_parameters', st.session_state.language))
        
        st.write(get_text('portfolio_parameters', st.session_state.language))
        pf_return = st.number_input(get_text('portfolio_return', st.session_state.language),
                                   min_value=0.0,
                                   max_value=1.0,
                                   value=0.1265,
                                   step=0.01,
                                   format="%.4f")
        
        pf_vol = st.slider(get_text('portfolio_volatility', st.session_state.language),
                          min_value=0.0,
                          max_value=0.50,
                          value=0.1525,
                          step=0.005)
        
        st.write(get_text('new_asset_parameters', st.session_state.language))
        new_vol = st.slider(get_text('new_asset_volatility', st.session_state.language),
                           min_value=0.0,
                           max_value=1.00,
                           value=0.60,
                           step=0.01)
        
        new_return = st.number_input(get_text('new_asset_return', st.session_state.language),
                                    min_value=-1.0,
                                    max_value=1.0,
                                    value=0.2988,
                                    step=0.01,
                                    format="%.4f")
        
        correlation = st.slider(get_text('correlation', st.session_state.language),
                              min_value=0.0,
                              max_value=1.0,
                              value=0.6,
                              step=0.05)

    # Right column
    with col2:
        # Compute optimal weights
        optimal_weights, port_return_fn, port_variance_fn = compute_optimal_weights(
            new_return, pf_return, new_vol, pf_vol, correlation
        )
        w_new, w_pf = optimal_weights

        # Break-even analysis
        st.write(f"### {get_text('break_even_analysis', st.session_state.language)}")
        st.write(get_text('break_even_condition', st.session_state.language))
        st.latex(r"\mu_a > S_p \cdot \rho \cdot \sigma_a")
        st.write(get_text('where', st.session_state.language))
        st.write(get_text('break_even_explanation', st.session_state.language))
        
        pf_sharpe = pf_return / pf_vol if pf_vol != 0 else 0
        break_even_return = pf_sharpe * correlation * new_vol
        st.write(get_text('break_even_return', st.session_state.language).format(pf_sharpe, break_even_return))

        # Efficient frontier plot
        st.write(f"### {get_text('efficient_frontier', st.session_state.language)}")
        fig = plot_efficient_frontier(
            new_return, pf_return, new_vol, pf_vol, correlation,
            optimal_weights, port_return_fn, port_variance_fn,
            st.session_state.language
        )
        st.pyplot(fig)

        # Optimal weights
        st.write(f"### {get_text('optimal_weights', st.session_state.language)}")
        st.write(f"{get_text('portfolio', st.session_state.language)}: {w_pf:.2%}")
        st.write(f"**{get_text('new_asset', st.session_state.language)}: {w_new:.2%}**")

        # Smith's analysis
        st.write(f"### {get_text('smith_analysis', st.session_state.language)}")
        st.write(get_text('smith_equation', st.session_state.language))
        st.latex(r"\frac{w_2}{w_1} = \frac{\sigma_1}{\sigma_2} \cdot \frac{SSR - \rho}{1 - SSR \cdot \rho}")
        st.write(get_text('where', st.session_state.language))
        st.write(get_text('smith_explanation', st.session_state.language))

        # Calculate Sharpe ratios
        sharpe_new = new_return/new_vol if new_vol != 0 else 0
        sharpe_pf = pf_return/pf_vol if pf_vol != 0 else 0
        SSR = sharpe_new/sharpe_pf if sharpe_pf != 0 else 0

        # Calculate Smith's ratio components
        vol_ratio = pf_vol/new_vol if new_vol != 0 else 0
        numerator = SSR - correlation
        denominator = 1 - (SSR * correlation)
        weight_ratio = vol_ratio * (numerator/denominator) if denominator != 0 else 0

        # Display calculations
        col_calc1, col_calc2 = st.columns(2)
        
        with col_calc1:
            st.write(get_text('sharpe_ratios', st.session_state.language))
            st.write(get_text('new_asset_sharpe', st.session_state.language).format(sharpe_new))
            st.write(get_text('portfolio_sharpe', st.session_state.language).format(sharpe_pf))
            st.write(get_text('ssr', st.session_state.language).format(SSR))

        with col_calc2:
            st.write(get_text('weight_ratio_components', st.session_state.language))
            st.write(get_text('volatility_ratio', st.session_state.language).format(vol_ratio))
            st.write(get_text('numerator', st.session_state.language).format(numerator))
            st.write(get_text('denominator', st.session_state.language).format(denominator))
            st.write(get_text('weight_ratio', st.session_state.language).format(weight_ratio))

        # Calculate and display Smith's weights
        w1_smith = 1 / (1 + weight_ratio) if weight_ratio != float('inf') else 0
        w2_smith = weight_ratio / (1 + weight_ratio) if weight_ratio != float('inf') else 1

        st.write(get_text('smith_optimal_weights', st.session_state.language))
        st.write(get_text('smith_portfolio', st.session_state.language).format(w1_smith))
        st.write(get_text('smith_new_asset', st.session_state.language).format(w2_smith))

if __name__ == "__main__":
    main()

### run in terminal: streamlit run 2_asset_markowitz_dashboard.py