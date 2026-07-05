
import json
import hmac
from pathlib import Path
from datetime import datetime, time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from api_iol import IOLClient, IOLAuthError, IOLApiError

st.set_page_config(page_title="BusaOptions Pro", layout="wide")

# =========================
# Seguridad
# =========================
def check_password():
    def get_app_password():
        try:
            if "APP_PASSWORD" in st.secrets:
                return str(st.secrets["APP_PASSWORD"])
            if "auth" in st.secrets and "password" in st.secrets["auth"]:
                return str(st.secrets["auth"]["password"])
        except Exception:
            pass
        local = Path("config/app_password.txt")
        if local.exists():
            return local.read_text(encoding="utf-8").strip()
        return ""

    expected = get_app_password()
    if not expected:
        st.warning("Falta configurar APP_PASSWORD en Streamlit Secrets o config/app_password.txt.")
        st.stop()

    if st.session_state.get("authenticated", False):
        return

    st.title("BusaOptions Pro")
    st.caption("Acceso privado")
    typed = st.text_input("Clave de acceso", type="password")
    if st.button("Entrar"):
        if hmac.compare_digest(typed, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Clave incorrecta.")
    st.stop()

check_password()

# =========================
# CSS mobile
# =========================
st.markdown("""
<style>
.block-container{padding-top:.75rem;padding-left:.55rem;padding-right:.55rem}
div[data-testid="stMetricValue"]{font-size:22px}
.prob-row{margin:18px 0 22px 0}
.prob-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.prob-name{font-weight:800;font-size:20px;color:#f9fafb}
.prob-val{font-weight:800;font-size:20px;color:#f9fafb}
.prob-bg{height:28px;background:#1f2937;border-radius:16px;overflow:hidden}
.prob-fill{height:28px;border-radius:16px}
.score-good{background:#12351f;color:#78ff9f;border-radius:10px;padding:4px 8px;font-weight:700}
.score-mid{background:#3b3112;color:#ffe27a;border-radius:10px;padding:4px 8px;font-weight:700}
.score-bad{background:#3b1616;color:#ff8c8c;border-radius:10px;padding:4px 8px;font-weight:700}
@media(max-width:768px){
  h1{font-size:1.45rem!important}
  h2,h3{font-size:1.08rem!important}
  button{width:100%}
  .prob-name,.prob-val{font-size:17px}
  .prob-bg,.prob-fill{height:24px}
}
</style>
""", unsafe_allow_html=True)

st.title("BusaOptions Pro 5.5")
st.caption("IOL + Black-Scholes + VI/VH + griegas + Score Busa + snapshot + acceso privado.")

TICKERS = {
    "GGAL": {"local": "GGAL.BA", "iol": "GGAL"},
    "YPF": {"local": "YPFD.BA", "iol": "YPFD"},
}

USAGE_FILE = Path("data/iol_api_usage.json")
SNAPSHOT_DIR = Path("data/snapshots")
FAVORITES_FILE = Path("data/favorites.json")
LIMIT = 25000

# =========================
# Persistencia local
# =========================
def current_month():
    return datetime.now().strftime("%Y-%m")

def load_usage():
    if not USAGE_FILE.exists():
        return {"month": current_month(), "calls": 0, "updates": 0, "last_update": None}
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {"month": current_month(), "calls": 0, "updates": 0, "last_update": None}
    if data.get("month") != current_month():
        return {"month": current_month(), "calls": 0, "updates": 0, "last_update": None}
    return data

def save_usage(data):
    USAGE_FILE.parent.mkdir(exist_ok=True)
    USAGE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def add_calls(n):
    data = load_usage()
    data["calls"] = int(data.get("calls", 0)) + int(n)
    data["updates"] = int(data.get("updates", 0)) + 1
    data["last_update"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    save_usage(data)

def reset_usage():
    save_usage({"month": current_month(), "calls": 0, "updates": 0, "last_update": None})

def snapshot_path(activo):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR / f"last_options_{activo}.json"

def save_snapshot(activo, raw):
    payload = {
        "activo": activo,
        "saved_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "raw": raw,
    }
    snapshot_path(activo).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def load_snapshot(activo):
    p = snapshot_path(activo)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_favorites():
    if not FAVORITES_FILE.exists():
        return []
    try:
        return json.loads(FAVORITES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_favorites(favs):
    FAVORITES_FILE.parent.mkdir(exist_ok=True)
    FAVORITES_FILE.write_text(json.dumps(sorted(set(favs)), indent=2, ensure_ascii=False), encoding="utf-8")

# =========================
# Datos y matemática
# =========================
@st.cache_data(ttl=180)
def get_hist(ticker, period):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def clean_num(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return np.nan
        if "," in x:
            x = x.replace(".", "").replace(",", ".")
    try:
        return float(x)
    except Exception:
        return np.nan

def bs_price(S, K, T, r, sigma, option_type):
    if min(S, K, T, sigma) <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks(S, K, T, r, sigma, option_type):
    if min(S, K, T, sigma) <= 0:
        return [np.nan] * 6
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    if option_type == "call":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return norm.cdf(d1), gamma, vega, theta, K*T*np.exp(-r*T)*norm.cdf(d2)/100, norm.cdf(d2)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    return -norm.cdf(-d1), gamma, vega, theta, -K*T*np.exp(-r*T)*norm.cdf(-d2)/100, norm.cdf(-d2)

def implied_vol(price, S, K, T, r, typ):
    try:
        return brentq(lambda sig: bs_price(S, K, T, r, sig, typ) - price, 0.001, 5.0)
    except Exception:
        return np.nan

def prob_data(close, horizon, lateral, lookback):
    rets = close.pct_change().dropna()
    lb = rets.tail(int(lookback))
    S = float(close.iloc[-1])
    hv = float(lb.std() * np.sqrt(252))
    mu = float(lb.mean() * 252)
    T = horizon / 252
    up = S * (1 + lateral)
    down = S * (1 - lateral)
    mean_log = np.log(S) + (mu - 0.5 * hv**2) * T
    sd = hv * np.sqrt(T)
    p_down = norm.cdf((np.log(down) - mean_log) / sd)
    p_up = 1 - norm.cdf((np.log(up) - mean_log) / sd)
    return {"VH": hv, "Sube": p_up, "Baja": p_down, "Lateral": max(0, 1-p_up-p_down), "Nivel suba": up, "Nivel baja": down}

def infer_tipo(symbol):
    s = str(symbol).upper()
    return "put" if ("GFGV" in s or "YPFV" in s or s.endswith("V")) else "call"

def normalize_options(raw):
    if isinstance(raw, dict):
        for key in ["opciones", "titulos", "data", "result", "items"]:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return pd.DataFrame()
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        titulo = item.get("titulo") if isinstance(item.get("titulo"), dict) else item
        cot = item.get("cotizacion") if isinstance(item.get("cotizacion"), dict) else item
        puntas = cot.get("puntas") if isinstance(cot.get("puntas"), dict) else {}
        simbolo = titulo.get("simbolo") or item.get("simbolo") or item.get("ticker") or item.get("descripcion") or ""
        strike = item.get("precioEjercicio") or item.get("strike") or titulo.get("precioEjercicio") or titulo.get("strike")
        if strike is None or pd.isna(strike):
            import re
            m = re.search(r"(\d{3,6})", str(simbolo))
            strike = m.group(1) if m else np.nan
        rows.append({
            "Ticker": str(simbolo).upper(),
            "Tipo": infer_tipo(simbolo),
            "Strike": clean_num(strike),
            "Compra": clean_num(puntas.get("precioCompra") or cot.get("precioCompra") or cot.get("compra") or item.get("compra")),
            "Venta": clean_num(puntas.get("precioVenta") or cot.get("precioVenta") or cot.get("venta") or item.get("venta")),
            "Último": clean_num(cot.get("ultimoPrecio") or cot.get("ultimo") or cot.get("precio") or item.get("ultimo")),
            "Volumen": clean_num(cot.get("volumen") or item.get("volumen")),
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["Strike"]).sort_values(["Tipo", "Strike", "Ticker"]) if not df.empty else df

def analyze(df, S, T, r, hv, p_up, p_down, mode):
    rows = []
    favorites = set(load_favorites())
    for _, row in df.iterrows():
        typ = str(row.get("Tipo", "call")).lower()
        K = clean_num(row.get("Strike"))
        compra = clean_num(row.get("Compra"))
        venta = clean_num(row.get("Venta"))
        ultimo = clean_num(row.get("Último"))
        if mode == "Promedio compra/venta":
            prima = np.nanmean([compra, venta]) if not (np.isnan(compra) and np.isnan(venta)) else ultimo
        elif mode == "Venta":
            prima = venta if not np.isnan(venta) else ultimo
        elif mode == "Compra":
            prima = compra if not np.isnan(compra) else ultimo
        else:
            prima = ultimo if not np.isnan(ultimo) else np.nanmean([compra, venta])

        theo = bs_price(S, K, T, r, hv, typ)
        iv = implied_vol(prima, S, K, T, r, typ) if not np.isnan(prima) else np.nan
        sig = iv if not np.isnan(iv) else hv
        delta, gamma, vega, theta, rho, prob_itm = greeks(S, K, T, r, sig, typ)
        intrinsic = max(S-K, 0) if typ == "call" else max(K-S, 0)
        extrinsic = prima - intrinsic if not np.isnan(prima) else np.nan
        diff = ((prima/theo)-1)*100 if not np.isnan(prima) and theo and theo > 0 else np.nan
        direction = p_up if typ == "call" else p_down

        score = np.nan
        state = "SIN PRECIO"
        if not np.isnan(prima):
            score = 50
            if not np.isnan(iv):
                if iv < hv - 0.10: score += 25
                elif iv < hv - 0.03: score += 12
                elif iv > hv + 0.15: score -= 25
                elif iv > hv + 0.07: score -= 12
            if 0.25 <= abs(delta) <= 0.60: score += 10
            if direction > 0.50: score += 12
            elif direction < 0.35: score -= 8
            score = max(0, min(100, score))
            if score >= 80: state = "🟢 OPORTUNIDAD"
            elif score >= 65: state = "🟡 INTERESANTE"
            else: state = "🔴 EVITAR / VIGILAR"

        rows.append({
            "Fav": "⭐" if row.get("Ticker") in favorites else "",
            "Ticker": row.get("Ticker"),
            "Tipo": typ.upper(),
            "Strike": K,
            "Compra": compra,
            "Venta": venta,
            "Último": ultimo,
            "Prima usada": prima,
            "Black-Scholes": theo,
            "Dif % vs BS": diff,
            "VI %": iv*100 if not np.isnan(iv) else np.nan,
            "VH %": hv*100,
            "Spread VI-VH": (iv-hv)*100 if not np.isnan(iv) else np.nan,
            "Intrínseco": intrinsic,
            "Extrínseco": extrinsic,
            "Delta": delta,
            "Gamma": gamma,
            "Vega x 1%": vega,
            "Theta diario": theta,
            "Prob. ITM %": prob_itm*100,
            "Prob. dirección %": direction*100,
            "Volumen": row.get("Volumen"),
            "Score Busa": score,
            "Estado": state,
        })
    return pd.DataFrame(rows)

def fmt(df):
    fmt_map = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    for c in ["Strike", "Score Busa", "Volumen"]:
        if c in fmt_map:
            fmt_map[c] = "{:.0f}"
    if "Delta" in fmt_map:
        fmt_map["Delta"] = "{:.3f}"
    if "Gamma" in fmt_map:
        fmt_map["Gamma"] = "{:.5f}"
    return df.style.format(fmt_map, na_rep="")

def prob_bar(label, pct, color):
    pct100 = max(0, min(100, pct * 100))
    st.markdown(f"""
    <div class="prob-row">
      <div class="prob-head"><span class="prob-name">{label}</span><span class="prob-val">{pct100:.1f}%</span></div>
      <div class="prob-bg"><div class="prob-fill" style="width:{pct100}%;background:{color};"></div></div>
    </div>
    """, unsafe_allow_html=True)

def market_status_text():
    now = datetime.now().time()
    # Referencia simple ARG: lunes a viernes 10:30-17:00 aprox.
    if datetime.now().weekday() < 5 and time(10,30) <= now <= time(17,0):
        return "🟢 Mercado posiblemente abierto"
    return "🟡 Mercado posiblemente cerrado"

# =========================
# Sidebar
# =========================
with st.sidebar:
    usage = load_usage()
    st.header("Actualizar")
    st.caption(market_status_text())
    st.metric("Consultas mes", f"{usage.get('calls', 0):,} / {LIMIT:,}")
    st.progress(min(1, usage.get("calls", 0) / LIMIT))
    if usage.get("last_update"):
        st.caption(f"Última API: {usage['last_update']}")

    activo = st.selectbox("Activo", ["GGAL", "YPF"])
    mode = st.selectbox("Prima usada", ["Promedio compra/venta", "Venta", "Compra", "Último"])

    with st.expander("Parámetros", expanded=False):
        period = st.selectbox("Histórico", ["6mo", "1y", "2y", "5y"], index=2)
        lookback = st.number_input("VH ruedas", 20, 252, 60, 5)
        horizon = st.number_input("Horizonte", 1, 120, 20)
        lateral = st.number_input("Lateral +/- %", 0.5, 30.0, 5.0, .5) / 100
        r = st.number_input("Tasa caución %", 0.0, 200.0, 20.2, .1) / 100
        days = st.number_input("Días vencimiento", 1, 365, 52)

    if st.button("🔄 Actualizar mercado (IOL)"):
        try:
            raw = IOLClient.from_config().get_options(TICKERS[activo]["iol"])
            st.session_state["raw_iol"] = raw
            st.session_state["options_df"] = normalize_options(raw)
            st.session_state["last_update"] = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")
            save_snapshot(activo, raw)
            add_calls(2)
            st.success("Mercado actualizado y snapshot guardado. Consumo estimado: 2 consultas.")
        except (FileNotFoundError, IOLAuthError, IOLApiError) as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🧮 Recalcular análisis"):
        if "options_df" not in st.session_state or st.session_state.get("options_df", pd.DataFrame()).empty:
            snap = load_snapshot(activo)
            if snap:
                st.session_state["raw_iol"] = snap["raw"]
                st.session_state["options_df"] = normalize_options(snap["raw"])
                st.session_state["last_update"] = snap.get("saved_at")
                st.success(f"Snapshot cargado: {snap.get('saved_at')}")
            else:
                st.warning("No hay datos cargados. Primero tocá Actualizar mercado.")
        st.rerun()

    if st.button("Reiniciar contador"):
        reset_usage()
        st.rerun()

    if st.button("Salir / bloquear"):
        st.session_state["authenticated"] = False
        st.rerun()

# =========================
# Carga base
# =========================
h = get_hist(TICKERS[activo]["local"], period)
if h.empty:
    st.error("No pude descargar histórico.")
    st.stop()

close = h["Close"].dropna()
S = float(close.iloc[-1])
prob = prob_data(close, int(horizon), lateral, int(lookback))
hv = prob["VH"]
T = days / 365

if "options_df" not in st.session_state or st.session_state.get("options_df", pd.DataFrame()).empty:
    snap = load_snapshot(activo)
    if snap:
        st.session_state["raw_iol"] = snap["raw"]
        st.session_state["options_df"] = normalize_options(snap["raw"])
        st.session_state["last_update"] = snap.get("saved_at")

df_options = st.session_state.get("options_df", pd.DataFrame())
analyzed = pd.DataFrame()
if not df_options.empty:
    analyzed = analyze(df_options, S, T, r, hv, prob["Sube"], prob["Baja"], mode)

# =========================
# UI
# =========================
tabs = st.tabs(["Dashboard", "Opciones", "Probabilidades", "Velas", "Favoritos"])

with tabs[0]:
    st.subheader(f"Dashboard {activo}")
    c1, c2 = st.columns(2)
    c1.metric("Precio", f"{S:,.2f}")
    c2.metric("VH", f"{hv*100:.1f}%")
    c3, c4 = st.columns(2)
    c3.metric("Prob. suba", f"{prob['Sube']:.1%}")
    c4.metric("Opciones", len(analyzed) if not analyzed.empty else 0)
    if "last_update" in st.session_state:
        st.caption(f"Último dato/snapshot: {st.session_state['last_update']}")
    st.caption(market_status_text())

    if not analyzed.empty:
        st.write("### Top oportunidades")
        top = analyzed.dropna(subset=["Score Busa"]).sort_values("Score Busa", ascending=False).head(5)
        st.dataframe(fmt(top), use_container_width=True)
    else:
        st.warning("Tocá Actualizar mercado para cargar opciones.")

with tabs[1]:
    st.subheader(f"Opciones {activo}")
    if "last_update" in st.session_state:
        st.caption(f"Última actualización/snapshot: {st.session_state['last_update']}")
    if analyzed.empty:
        st.warning("Sin cadena cargada. Tocá Actualizar mercado.")
    else:
        show_only = st.selectbox("Filtro rápido", ["Todas", "Solo oportunidades", "Score >= 70", "Favoritas"])
        view = analyzed.copy()
        if show_only == "Solo oportunidades":
            view = view[view["Score Busa"] >= 80]
        elif show_only == "Score >= 70":
            view = view[view["Score Busa"] >= 70]
        elif show_only == "Favoritas":
            view = view[view["Fav"] == "⭐"]

        st.write("### Ranking")
        st.dataframe(fmt(view.dropna(subset=["Score Busa"]).sort_values("Score Busa", ascending=False).head(30)), use_container_width=True)

        st.write("### CALLS")
        st.dataframe(fmt(view[view["Tipo"] == "CALL"].sort_values("Strike")), use_container_width=True)

        st.write("### PUTS")
        st.dataframe(fmt(view[view["Tipo"] == "PUT"].sort_values("Strike")), use_container_width=True)

with tabs[2]:
    st.subheader("Probabilidades")
    prob_bar("Sube", prob["Sube"], "#15803d")
    prob_bar("Baja", prob["Baja"], "#b91c1c")
    prob_bar("Lateral", prob["Lateral"], "#ca8a04")
    c1, c2 = st.columns(2)
    c1.metric("Nivel suba", f"{prob['Nivel suba']:,.2f}")
    c2.metric("Nivel baja", f"{prob['Nivel baja']:,.2f}")
    st.metric("Precio base", f"{S:,.2f}")

with tabs[3]:
    st.subheader("Velas")
    if st.button("🔁 Reset vista velas"):
        st.session_state["chart_revision"] = st.session_state.get("chart_revision", 0) + 1

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"], name="Velas"))
    fig.add_trace(go.Scatter(x=h.index, y=ema20, name="EMA20", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=h.index, y=ema50, name="EMA50", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=h.index, y=ema200, name="EMA200", line=dict(width=1.4)))
    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
        dragmode="zoom",
        uirevision=st.session_state.get("chart_revision", 0),
        margin=dict(l=10, r=10, t=30, b=20),
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "doubleClick": "reset",
            "modeBarButtonsToAdd": ["pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
            "displaylogo": False,
        },
    )
    st.caption("Doble clic resetea. Toolbar: zoom, pan, autoscale y reset.")

with tabs[4]:
    st.subheader("Favoritos")
    favs = load_favorites()
    st.caption("Guardá tickers que querés seguir. Ejemplo: GFGC8600AG")
    new_fav = st.text_input("Agregar favorito")
    if st.button("Agregar favorito"):
        if new_fav.strip():
            favs.append(new_fav.strip().upper())
            save_favorites(favs)
            st.rerun()
    if favs:
        st.write("### Lista")
        st.write(", ".join(favs))
        remove = st.selectbox("Quitar favorito", [""] + favs)
        if st.button("Quitar seleccionado") and remove:
            favs = [x for x in favs if x != remove]
            save_favorites(favs)
            st.rerun()
    else:
        st.info("Todavía no hay favoritos.")

with st.expander("Debug IOL", expanded=False):
    if "raw_iol" in st.session_state:
        st.json(st.session_state["raw_iol"])
    else:
        st.info("Sin respuesta cruda.")
