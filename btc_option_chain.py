import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import yfinance as yf

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Crypto Option Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# DICCIONARIO DE ACTIVOS
# ============================================================================
CRYPTO_ASSETS = {
    "Bitcoin (BTC)": {"ticker": "BTC-USD", "icon": "₿"},
    "Ethereum (ETH)": {"ticker": "ETH-USD", "icon": "Ξ"},
    "Solana (SOL)": {"ticker": "SOL-USD", "icon": "◎"},
    "XRP (XRP)": {"ticker": "XRP-USD", "icon": "✕"}
}

# ============================================================================
# MODELO BLACK-SCHOLES
# ============================================================================
class BlackScholes:
    """
    Implementación del modelo Black-Scholes para pricing de opciones.
    """
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def call_delta(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    @staticmethod
    def put_delta(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ============================================================================
# FUNCIONES PARA YAHOO FINANCE (ADAPTADAS PARA MÚLTIPLES ACTIVOS)
# ============================================================================
@st.cache_data(ttl=30)  # Cache por 30 segundos
def get_crypto_price(ticker):
    """
    Obtiene el precio actual del ticker seleccionado usando Yahoo Finance.
    """
    try:
        # Descarga datos del ticker seleccionado
        asset = yf.Ticker(ticker)
        
        # Obtiene información en tiempo real
        info = asset.info
        
        # Obtiene datos históricos del último día para calcular cambio
        hist = asset.history(period="1d")
        
        if hist.empty:
            st.error(f"❌ No se pudieron obtener datos históricos para {ticker}")
            return None, None, None
        
        # Precio actual
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if current_price is None:
            # Fallback: usar el último precio de cierre
            current_price = hist['Close'].iloc[-1]
        
        # Cambio en 24h
        previous_close = info.get('previousClose', hist['Open'].iloc[0])
        change_24h = ((current_price - previous_close) / previous_close) * 100
        
        # Volumen
        volume = info.get('volume', hist['Volume'].iloc[-1])
        
        return current_price, change_24h, volume
        
    except Exception as e:
        st.error(f"❌ Error al obtener datos de Yahoo Finance ({ticker}): {str(e)}")
        return None, None, None

@st.cache_data(ttl=300)  # Cache por 5 minutos
def calculate_historical_volatility(ticker, period="30d"):
    """
    Calcula la volatilidad histórica del ticker seleccionado.
    """
    try:
        asset = yf.Ticker(ticker)
        hist = asset.history(period=period)
        
        if hist.empty:
            return 0.65  # Volatilidad por defecto
        
        # Calcula retornos logarítmicos
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        
        # Volatilidad diaria
        daily_vol = log_returns.std()
        
        # Anualiza (asumiendo 365 días de trading en crypto)
        annual_vol = daily_vol * np.sqrt(365)
        
        return annual_vol
        
    except Exception as e:
        st.warning(f"⚠️ No se pudo calcular volatilidad histórica: {str(e)}")
        return 0.65  # Valor por defecto

# ============================================================================
# GENERACIÓN DE OPTION CHAIN
# ============================================================================
def generate_option_chain(spot_price, iv, days_to_expiry, risk_free_rate, num_strikes):
    # Tiempo hasta expiración en años
    T = days_to_expiry / 365.0
    
    # Espaciado dinámico basado en el precio (2.5% del spot)
    # Esto funciona tanto para BTC (90k -> salto de 2250) como para XRP (2.5 -> salto de 0.06)
    strike_spacing = spot_price * 0.025
    strikes = []
    
    for i in range(-num_strikes, num_strikes + 1):
        strike = spot_price + (i * strike_spacing)
        # Redondeo inteligente según el precio del activo
        if spot_price < 10:
            strikes.append(round(strike, 4))
        else:
            strikes.append(round(strike, 2))
    
    # Calcula métricas para cada strike
    chain_data = []
    
    for K in strikes:
        # Calls
        call_price = BlackScholes.call_price(spot_price, K, T, risk_free_rate, iv)
        call_delta = BlackScholes.call_delta(spot_price, K, T, risk_free_rate, iv)
        call_prob_itm = abs(call_delta) * 100 
        
        # Puts
        put_price = BlackScholes.put_price(spot_price, K, T, risk_free_rate, iv)
        put_delta = BlackScholes.put_delta(spot_price, K, T, risk_free_rate, iv)
        put_prob_itm = abs(put_delta) * 100 
        
        # Gamma
        gamma_val = BlackScholes.gamma(spot_price, K, T, risk_free_rate, iv)
        
        chain_data.append({
            "Call Prob. ITM": call_prob_itm,
            "Call Price": call_price,
            "Call Delta": call_delta,
            "STRIKE": K,
            "Put Price": put_price,
            "Put Delta": put_delta,
            "Put Prob. ITM": put_prob_itm,
            "Gamma": gamma_val,
            "ITM_Call": spot_price > K,
            "ITM_Put": spot_price < K
        })
    
    df = pd.DataFrame(chain_data)
    return df

# ============================================================================
# FUNCIÓN PARA ESTILIZAR EL DATAFRAME
# ============================================================================
def style_option_chain(df, spot_price):
    display_cols = ["Call Prob. ITM", "Call Price", "STRIKE", "Put Price", "Put Prob. ITM"]
    
    def highlight_itm(row):
        styles = [''] * len(row)
        
        # Call ITM
        if spot_price > row['STRIKE']:
            styles[0] = 'background-color: #d4edda'
            styles[1] = 'background-color: #d4edda'
        
        # Put ITM
        if spot_price < row['STRIKE']:
            styles[3] = 'background-color: #f8d7da'
            styles[4] = 'background-color: #f8d7da'
        
        # Resalta el ATM (aprox 1.5% de diferencia)
        if abs(row['STRIKE'] - spot_price) < spot_price * 0.015:
            styles[2] = 'background-color: #fff3cd; font-weight: bold'
        
        return styles
    
    styled_df = df[display_cols].style.apply(highlight_itm, axis=1, subset=display_cols)
    
    # Formato de moneda inteligente
    currency_fmt = "${:,.2f}" if spot_price > 10 else "${:,.4f}"
    
    styled_df = styled_df.format({
        "Call Prob. ITM": "{:.1f}%",
        "Call Price": currency_fmt,
        "STRIKE": currency_fmt,
        "Put Price": currency_fmt,
        "Put Prob. ITM": "{:.1f}%"
    })
    
    return styled_df

# ============================================================================
# INTERFAZ DE STREAMLIT
# ============================================================================
def main():
    # ========================================================================
    # SIDEBAR: Configuración y Selección de Activo
    # ========================================================================
    st.sidebar.header("⚙️ Configuración")
    
    # 1. SELECTOR DE CRIPTOMONEDA
    selected_asset_name = st.sidebar.selectbox(
        "Selecciona Activo",
        options=list(CRYPTO_ASSETS.keys()),
        index=0
    )
    
    # Obtener datos del activo seleccionado
    asset_data = CRYPTO_ASSETS[selected_asset_name]
    ticker = asset_data["ticker"]
    icon = asset_data["icon"]
    
    st.sidebar.markdown("---")
    
    # Selector de modo de volatilidad
    vol_mode = st.sidebar.radio(
        "Modo de Volatilidad",
        ["Manual", "Histórica (30d)", "Histórica (90d)"],
        help="Elige entre volatilidad manual o calculada desde datos históricos"
    )
    
    # Parámetros del modelo
    if vol_mode == "Manual":
        iv = st.sidebar.slider(
            "Volatilidad Implícita (IV)",
            min_value=10,
            max_value=200, # Aumenté el rango para Altcoins
            value=65,
            step=5,
            help="Volatilidad anualizada en %"
        ) / 100.0
    elif vol_mode == "Histórica (30d)":
        with st.spinner(f"Calculando volatilidad histórica de {ticker}..."):
            iv = calculate_historical_volatility(ticker, "30d")
        st.sidebar.info(f"📊 Vol. Histórica 30d ({ticker}): **{iv*100:.1f}%**")
    else:  # 90d
        with st.spinner(f"Calculando volatilidad histórica de {ticker}..."):
            iv = calculate_historical_volatility(ticker, "90d")
        st.sidebar.info(f"📊 Vol. Histórica 90d ({ticker}): **{iv*100:.1f}%**")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parámetros de Opciones")
    
    days_to_expiry = st.sidebar.slider(
        "Días hasta Expiración",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )
    
    risk_free_rate = st.sidebar.slider(
        "Tasa Libre de Riesgo (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Tasa anualizada (ej. T-Bills)"
    ) / 100.0
    
    num_strikes = st.sidebar.slider(
        "Strikes a cada lado del ATM",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.success(
        "✅ **Yahoo Finance**\n\n"
        "Sin API key, gratis e ilimitado. "
    )
    
    # ========================================================================
    # TÍTULO Y OBTENCIÓN DE DATOS
    # ========================================================================
    st.title(f"{icon} {selected_asset_name.split(' ')[0]} Option Chain Dashboard")
    st.markdown("**Powered by Yahoo Finance + Black-Scholes Model**")
    st.markdown("---")

    with st.spinner(f"Obteniendo precio de {ticker} desde Yahoo Finance..."):
        spot_price, change_24h, volume = get_crypto_price(ticker)
    
    if spot_price is None:
        st.error(f"❌ No se pudo obtener el precio de {selected_asset_name}. Verifica tu conexión.")
        st.info("🔄 Intenta recargar la página o hacer click en 'Actualizar Datos'")
        return
    
    # ========================================================================
    # MÉTRICAS PRINCIPALES
    # ========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Formato condicional para precios pequeños (como XRP)
    price_fmt = "${:,.4f}" if spot_price < 10 else "${:,.2f}"
    
    with col1:
        st.metric(
            f"💰 Precio {ticker.split('-')[0]}", 
            price_fmt.format(spot_price),
            f"{change_24h:+.2f}%" if change_24h else None
        )
    
    with col2:
        st.metric("📊 IV Actual", f"{iv*100:.1f}%")
    
    with col3:
        expiry_date = datetime.now() + timedelta(days=days_to_expiry)
        st.metric("📅 Expiración", expiry_date.strftime("%d/%m/%Y"))
    
    with col4:
        st.metric("⏱️ Días a Exp.", days_to_expiry)
    
    with col5:
        if volume:
            st.metric("📦 Volumen 24h", f"${volume/1e9:.2f}B")
        else:
            st.metric("📦 Volumen 24h", "N/A")
    
    st.markdown("---")
    
    # ========================================================================
    # GENERACIÓN Y VISUALIZACIÓN DE LA CADENA
    # ========================================================================
    with st.spinner("Generando Option Chain con Black-Scholes..."):
        option_chain = generate_option_chain(
            spot_price=spot_price,
            iv=iv,
            days_to_expiry=days_to_expiry,
            risk_free_rate=risk_free_rate,
            num_strikes=num_strikes
        )
    
    st.subheader(f"📈 Cadena de Opciones ({ticker.split('-')[0]})")
    st.markdown(
        "**Calls** (izquierda) | **Strikes** (centro) | **Puts** (derecha)  \n"
        "🟢 Verde = ITM | ATM resaltado en amarillo"
    )
    
    # Aplica estilos y muestra la tabla
    styled_chain = style_option_chain(option_chain, spot_price)
    st.dataframe(styled_chain, use_container_width=True, height=600)
    
    # ========================================================================
    # ANÁLISIS DE GREEKS
    # ========================================================================
    with st.expander("📊 Análisis de Greeks (ATM)"):
        # Encuentra el strike ATM más cercano
        atm_idx = (option_chain['STRIKE'] - spot_price).abs().idxmin()
        atm_row = option_chain.iloc[atm_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Call Delta", f"{atm_row['Call Delta']:.4f}")
            st.caption("Sensibilidad al precio (~Prob ITM)")
        
        with col2:
            st.metric("Put Delta", f"{atm_row['Put Delta']:.4f}")
            st.caption("Sensibilidad al precio (~Prob ITM)")
        
        with col3:
            st.metric("Gamma", f"{atm_row['Gamma']:.6f}")
            st.caption("Tasa de cambio del Delta")
    
    # ========================================================================
    # INFORMACIÓN ADICIONAL
    # ========================================================================
    with st.expander("ℹ️ Detalles Técnicos del Modelo"):
        st.markdown(f"""
        **Modelo de Pricing:** Black-Scholes para Opciones Europeas
        
        **Parámetros Actuales:**
        - Spot Price (S): {price_fmt.format(spot_price)}
        - Volatilidad Implícita (σ): {iv*100:.1f}%
        - Tiempo a Expiración (T): {days_to_expiry/365:.4f} años
        - Tasa Libre de Riesgo (r): {risk_free_rate*100:.1f}%
        
        **Nota:** Los strikes se generan automáticamente a un ±2.5% de distancia relativa del precio spot actual.
        """)
    
    # Footer
    st.markdown("---")
    st.caption(
        "Desarrollado con Streamlit | Datos de Yahoo Finance | "
        f"Modelo Black-Scholes | Última actualización: {datetime.now().strftime('%H:%M:%S')}"
    )

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    main()