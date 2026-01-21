import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date, timedelta
import ccxt
import time
from supabase import create_client, Client
import bcrypt

# ============================================================================
# 1. CONFIGURACIÓN Y CONEXIÓN A BASE DE DATOS
# ============================================================================
st.set_page_config(page_title="QuantProb | Pro Terminal", page_icon="🚀", layout="wide")

# Inicializar conexión a Supabase
try:
    supabase_url = st.secrets["supabase"]["url"]
    supabase_key = st.secrets["supabase"]["key"]
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error("Error configurando Base de Datos. Revisa tus st.secrets.")
    st.stop()

# Estilos CSS
st.markdown("""
<style>
    .stDataFrame { font-size: 14px; }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .big-button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SISTEMA DE AUTENTICACIÓN (LOGIN / REGISTRO)
# ============================================================================

def hash_pass(password):
    """Cifra la contraseña"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_pass(password, hashed):
    """Verifica la contraseña"""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def login_user(email, password):
    try:
        # Consulta a Supabase
        response = supabase.table('users').select("*").eq('email', email).execute()
        if len(response.data) > 0:
            user = response.data[0]
            if check_pass(password, user['password']):
                st.session_state.user = user
                st.success(f"¡Bienvenido de nuevo, {user['name']}!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")
        else:
            st.error("El usuario no existe.")
    except Exception as e:
        st.error(f"Error de conexión: {e}")

def register_user(email, password, name):
    try:
        # Verificar si ya existe
        check = supabase.table('users').select("email").eq('email', email).execute()
        if len(check.data) > 0:
            st.warning("Este email ya está registrado.")
            return

        # Insertar nuevo usuario (is_pro = False por defecto)
        hashed_pw = hash_pass(password)
        new_user = {
            "email": email,
            "password": hashed_pw,
            "name": name,
            "is_pro": False
        }
        supabase.table('users').insert(new_user).execute()
        st.success("¡Cuenta creada con éxito! Ahora inicia sesión.")
    except Exception as e:
        st.error(f"Error al registrar: {e}")

# --- PANTALLA DE ACCESO ---
if 'user' not in st.session_state:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.title("🔐 QuantProb Access")
        tab_login, tab_reg = st.tabs(["Iniciar Sesión", "Crear Cuenta"])
        
        with tab_login:
            l_email = st.text_input("Email", key="l_email")
            l_pass = st.text_input("Contraseña", type="password", key="l_pass")
            if st.button("Entrar", use_container_width=True):
                login_user(l_email, l_pass)
        
        with tab_reg:
            r_name = st.text_input("Nombre", key="r_name")
            r_email = st.text_input("Email", key="r_email")
            r_pass = st.text_input("Contraseña", type="password", key="r_pass")
            if st.button("Registrarse Gratis", use_container_width=True):
                if r_email and r_pass:
                    register_user(r_email, r_pass, r_name)
                else:
                    st.warning("Rellena todos los campos")
    st.stop() # DETIENE LA EJECUCIÓN SI NO HAY LOGIN

# ============================================================================
# 3. LÓGICA DE NEGOCIO (VERIFICACIÓN PRO)
# ============================================================================
user = st.session_state.user
IS_PRO = user.get('is_pro', False)
USER_NAME = user.get('name', 'Trader')

# --- SIDEBAR DEL USUARIO ---
with st.sidebar:
    st.write(f"👤 **{USER_NAME}**")
    
    if IS_PRO:
        st.success("ESTADO: 🏆 MIEMBRO PRO")
    else:
        st.warning("ESTADO: 🧊 PLAN GRATUITO")
        st.markdown("---")
        st.markdown("### 🚀 Sube de Nivel")
        st.caption("Desbloquea Solana, Ethereum y Modo Touch.")
        
        # BOTÓN DE STRIPE
        stripe_url = st.secrets["stripe"]["link"]
        st.link_button("👉 Activar PRO ($29)", stripe_url, type="primary")
        st.info("ℹ️ Tras pagar, tu cuenta se activará en <24h.")
    
    if st.button("Cerrar Sesión"):
        st.session_state.pop('user')
        st.rerun()
    st.divider()

# ============================================================================
# 4. APLICACIÓN PRINCIPAL (Binance + Black-Scholes)
# ============================================================================

# Configuración Binance
exchange = ccxt.binance({'enableRateLimit': True})

# Activos Disponibles (LÓGICA DE BLOQUEO)
FULL_ASSETS = {
    "Bitcoin (BTC)": {"ticker": "BTC/USDT", "icon": "₿", "vol": 55},
    "Ethereum (ETH)": {"ticker": "ETH/USDT", "icon": "Ξ", "vol": 65},
    "Solana (SOL)": {"ticker": "SOL/USDT", "icon": "◎", "vol": 85},
    "XRP (XRP)": {"ticker": "XRP/USDT", "icon": "✕", "vol": 95},
}

if IS_PRO:
    available_assets = list(FULL_ASSETS.keys())
else:
    available_assets = ["Bitcoin (BTC)"] # Gratis solo ven BTC

# --- INTERFAZ ---
st.title("Terminal de Análisis Cuantitativo")

# Selector de Activos
col_sel, col_date = st.columns(2)
with col_sel:
    selected_name = st.selectbox("Seleccionar Activo", available_assets)
    
    # Upsell si intentan buscar más (UX Hack)
    if not IS_PRO:
        st.caption("🔒 ETH y SOL bloqueados. [Hazte PRO](#) para desbloquear.")

with col_date:
    expiry = st.date_input("Vencimiento", date.today() + timedelta(days=30))

# Datos del activo seleccionado
asset = FULL_ASSETS[selected_name]
symbol = asset['ticker']
default_vol = asset['vol']

# Funciones de Cálculo (Black-Scholes simplificado para el ejemplo)
@st.cache_data(ttl=10)
def get_price(sym):
    try:
        t = exchange.fetch_ticker(sym)
        return t['last'], t['percentage']
    except: return None, None

price, change = get_price(symbol)

if not price:
    st.error("Error conectando con Binance API.")
    st.stop()

# Renderizado de Métricas
m1, m2, m3 = st.columns(3)
m1.metric("Precio Spot", f"${price:,.2f}", f"{change:.2f}%")
m2.metric("Volatilidad Base", f"{default_vol}%")
m3.metric("Días Restantes", (expiry - date.today()).days)

# --- TABLA DE PROBABILIDADES ---
st.subheader(f"Matriz de Probabilidad: {selected_name}")

# (Aquí pegamos la lógica matemática que ya tenías)
# Generamos datos dummy basados en la fórmula real para visualizar
days = (expiry - date.today()).days
sigma = default_vol / 100
T = days / 365
r = 0.05
strikes = [price * (1 + i*0.02) for i in range(-5, 6)]
data = []

for k in strikes:
    d2 = (np.log(price/k) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    prob_call = norm.cdf(d2) * 100
    
    # MODIFICADOR PRO: MODO TOUCH
    # Solo los PRO ven la probabilidad de "Touch" real
    if IS_PRO:
        prob_touch = min(prob_call * 2, 99) # Simple estimación One-Touch
    else:
        prob_touch = 0 # Oculto
        
    data.append({
        "Strike": k,
        "Prob Cierre (Expiry)": prob_call,
        "Prob Tocar (Touch) [PRO]": prob_touch if IS_PRO else 0
    })

df = pd.DataFrame(data)

# Configuración de columnas
col_config = {
    "Strike": st.column_config.NumberColumn(format="$%.2f"),
    "Prob Cierre (Expiry)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
}

if IS_PRO:
    col_config["Prob Tocar (Touch) [PRO]"] = st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)
else:
    # Ocultar o mostrar bloqueado
    pass 

st.dataframe(df, use_container_width=True, column_config=col_config)

if not IS_PRO:
    st.info("💡 **Nota:** La columna 'Probabilidad de Tocar' está oculta en el plan gratuito.")