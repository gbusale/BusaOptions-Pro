
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


# =========================
# Compatibilidad API IOL
# =========================
if not hasattr(IOLClient, "get_quote"):
    def _iol_get_quote(self, simbolo: str, mercado: str = "bCBA"):
        return self.get(f"/api/v2/{mercado}/Titulos/{simbolo}/Cotizacion")
    IOLClient.get_quote = _iol_get_quote

if not hasattr(IOLClient, "get_history"):
    def _iol_get_history(self, simbolo: str, fecha_desde: str, fecha_hasta: str, ajustada: str = "SinAjustar", mercado: str = "bCBA"):
        return self.get(f"/api/v2/{mercado}/Titulos/{simbolo}/Cotizacion/seriehistorica/{fecha_desde}/{fecha_hasta}/{ajustada}")
    IOLClient.get_history = _iol_get_history

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

st.title("BusaOptions Pro 9.1.2")
st.caption("IOL + Black-Scholes + Busa AI + Advisor 9.1 + Learning claro.")

TICKERS = {
    "GGAL": {"local": "GGAL.BA", "iol": "GGAL"},
    "YPF": {"local": "YPFD.BA", "iol": "YPFD"},
}

USAGE_FILE = Path("data/iol_api_usage.json")
SNAPSHOT_DIR = Path("data/snapshots")
FAVORITES_FILE = Path("data/favorites.json")
LEARNING_FILE = Path("data/learning_log.csv")
PREDICTIONS_FILE = Path("data/predictions_log.csv")
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

def save_snapshot(activo, raw, spot=None, quote_raw=None):
    payload = {
        "activo": activo,
        "saved_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "spot": spot,
        "quote_raw": quote_raw,
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

def load_learning():
    if not LEARNING_FILE.exists():
        return pd.DataFrame(columns=[
            "Fecha",
            "Activo",
            "Precio inicial",
            "Precio cierre",
            "Variación %",
            "Prob. suba",
            "Prob. baja",
            "Prob. lateral",
            "Predicción",
            "Resultado",
            "Acierto",
        ])
    try:
        return pd.read_csv(LEARNING_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Fecha",
            "Activo",
            "Precio inicial",
            "Precio cierre",
            "Variación %",
            "Prob. suba",
            "Prob. baja",
            "Prob. lateral",
            "Predicción",
            "Resultado",
            "Acierto",
        ])

def save_learning(df):
    LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEARNING_FILE, index=False, encoding="utf-8-sig")

def dominant_prediction(prob_dict):
    values = {
        "Sube": float(prob_dict.get("Sube", 0)),
        "Baja": float(prob_dict.get("Baja", 0)),
        "Lateral": float(prob_dict.get("Lateral", 0)),
    }
    return max(values, key=values.get)

def technical_features(hist_df):
    """
    Calcula indicadores técnicos simples para registrar contexto del día.
    No usa estas variables todavía para predecir, pero quedan guardadas
    para entrenar Busa AI más adelante.
    """
    out = {
        "RSI14": np.nan,
        "Retorno 1d %": np.nan,
        "Retorno 5d %": np.nan,
        "Dist EMA20 %": np.nan,
        "Dist EMA50 %": np.nan,
        "ATR14 %": np.nan,
        "Volumen relativo": np.nan,
    }
    try:
        c = hist_df["Close"].dropna()
        if len(c) < 20:
            return out

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()

        high = hist_df["High"]
        low = hist_df["Low"]
        prev_close = hist_df["Close"].shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        vol_rel = np.nan
        if "Volume" in hist_df.columns and hist_df["Volume"].dropna().shape[0] >= 20:
            vol = hist_df["Volume"].dropna()
            vol_rel = float(vol.iloc[-1] / vol.tail(20).mean()) if vol.tail(20).mean() else np.nan

        out.update({
            "RSI14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else np.nan,
            "Retorno 1d %": float(c.pct_change(1).iloc[-1] * 100),
            "Retorno 5d %": float(c.pct_change(5).iloc[-1] * 100),
            "Dist EMA20 %": float((c.iloc[-1] / ema20.iloc[-1] - 1) * 100),
            "Dist EMA50 %": float((c.iloc[-1] / ema50.iloc[-1] - 1) * 100),
            "ATR14 %": float(atr.iloc[-1] / c.iloc[-1] * 100) if not pd.isna(atr.iloc[-1]) else np.nan,
            "Volumen relativo": vol_rel,
        })
    except Exception:
        pass
    return out

def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame(columns=[
            "Fecha señal", "Activo", "Precio inicial", "Prob. suba", "Prob. baja", "Prob. lateral",
            "Predicción", "VH %", "RSI14", "Retorno 1d %", "Retorno 5d %",
            "Dist EMA20 %", "Dist EMA50 %", "ATR14 %", "Volumen relativo",
            "Evaluada", "Fecha evaluación", "Precio cierre", "Variación %", "Resultado", "Acierto"
        ])
    try:
        return pd.read_csv(PREDICTIONS_FILE)
    except Exception:
        return pd.DataFrame()

def save_predictions(df):
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False, encoding="utf-8-sig")

def learning_factor_for_asset(activo):
    """
    Primer ajuste adaptativo:
    - Si viene acertando bien, aumenta suavemente la probabilidad dominante.
    - Si viene fallando, la modera.
    Requiere al menos 5 señales evaluadas para activarse.
    """
    df = load_predictions()
    if df.empty or "Activo" not in df.columns or "Acierto" not in df.columns:
        return 1.0, 0, np.nan

    d = df[(df["Activo"].astype(str) == activo) & (pd.to_numeric(df.get("Evaluada", 0), errors="coerce").fillna(0).astype(int) == 1)].copy()
    if d.empty:
        return 1.0, 0, np.nan

    d = d.tail(20)
    n = len(d)
    acc = pd.to_numeric(d["Acierto"], errors="coerce").mean()

    if n < 5 or pd.isna(acc):
        return 1.0, n, acc

    if acc >= 0.65:
        return 1.15, n, acc
    if acc >= 0.58:
        return 1.08, n, acc
    if acc <= 0.40:
        return 0.85, n, acc
    if acc <= 0.48:
        return 0.92, n, acc
    return 1.0, n, acc

def apply_learning_to_probabilities(prob_dict, activo):
    """
    Ajusta la probabilidad dominante y renormaliza.
    Guarda datos de control para mostrar en pantalla.
    """
    factor, n, acc = learning_factor_for_asset(activo)

    probs = np.array([
        float(prob_dict.get("Sube", 0)),
        float(prob_dict.get("Baja", 0)),
        float(prob_dict.get("Lateral", 0)),
    ], dtype=float)

    if probs.sum() <= 0:
        return prob_dict

    idx = int(np.argmax(probs))
    original = probs.copy()
    probs[idx] *= factor
    probs = probs / probs.sum()

    out = dict(prob_dict)
    out["Sube base"] = original[0]
    out["Baja base"] = original[1]
    out["Lateral base"] = original[2]
    out["Sube"] = probs[0]
    out["Baja"] = probs[1]
    out["Lateral"] = probs[2]
    out["Learning factor"] = factor
    out["Learning n"] = n
    out["Learning accuracy"] = acc
    return out

def next_available_close(hist_df, signal_date):
    """
    Busca el primer cierre posterior a la fecha de señal.
    Sirve para evaluar automáticamente aunque haya fin de semana o feriado.
    """
    try:
        d0 = pd.to_datetime(signal_date).date()
        tmp = hist_df.copy()
        tmp = tmp.dropna(subset=["Close"])
        tmp_dates = pd.to_datetime(tmp.index).date
        mask = tmp_dates > d0
        if not mask.any():
            return None, None
        idx = np.where(mask)[0][0]
        eval_date = pd.to_datetime(tmp.index[idx]).strftime("%Y-%m-%d")
        close_price = float(tmp["Close"].iloc[idx])
        return eval_date, close_price
    except Exception:
        return None, None

def evaluate_pending_predictions_auto(activo, hist_df, lateral_threshold):
    """
    Evalúa señales pendientes usando el primer cierre disponible posterior
    a la fecha de señal. No requiere cargar precio manual.
    """
    df = load_predictions()
    if df.empty or "Evaluada" not in df.columns:
        return 0

    count = 0
    for idx, row in df.iterrows():
        if str(row.get("Activo")) != activo:
            continue
        try:
            evaluated = int(float(row.get("Evaluada", 0)))
        except Exception:
            evaluated = 0
        if evaluated == 1:
            continue

        signal_date = row.get("Fecha señal")
        eval_date, close_price = next_available_close(hist_df, signal_date)
        if eval_date is None or close_price is None:
            continue

        initial = clean_num(row.get("Precio inicial"))
        if np.isnan(initial) or initial <= 0:
            continue

        variation = close_price / initial - 1
        result = "Sube" if variation > lateral_threshold else "Baja" if variation < -lateral_threshold else "Lateral"
        pred = str(row.get("Predicción"))
        hit = int(result == pred)

        df.loc[idx, "Evaluada"] = 1
        df.loc[idx, "Fecha evaluación"] = eval_date
        df.loc[idx, "Precio cierre"] = close_price
        df.loc[idx, "Variación %"] = variation * 100
        df.loc[idx, "Resultado"] = result
        df.loc[idx, "Acierto"] = hit
        count += 1

    if count:
        save_predictions(df)
    return count


def learning_status_text(row):
    try:
        evaluated = int(float(row.get("Evaluada", 0)))
    except Exception:
        evaluated = 0
    if evaluated != 1:
        return "⏳ Pendiente"
    try:
        hit = int(float(row.get("Acierto", 0)))
    except Exception:
        hit = 0
    return "✅ Acertó" if hit == 1 else "❌ Falló"

def add_learning_status_columns(df):
    if df is None or df.empty:
        return df
    d = df.copy()
    d["Estado aprendizaje"] = d.apply(learning_status_text, axis=1)
    def msg(row):
        if row.get("Estado aprendizaje") == "⏳ Pendiente":
            return f"Predijo {row.get('Predicción')} y todavía no hay cierre posterior evaluado."
        return f"Predijo {row.get('Predicción')}; resultado real {row.get('Resultado')}; variación {clean_num(row.get('Variación %')):.2f}%."
    d["Lectura"] = d.apply(msg, axis=1)
    return d

def evaluate_single_prediction_manual(row_index, close_price, eval_date, lateral_threshold):
    df = load_predictions()
    if df.empty or row_index not in df.index:
        return False, "No encontré la señal."

    initial = clean_num(df.loc[row_index, "Precio inicial"])
    if np.isnan(initial) or initial <= 0:
        return False, "Precio inicial inválido."

    variation = close_price / initial - 1
    result = "Sube" if variation > lateral_threshold else "Baja" if variation < -lateral_threshold else "Lateral"
    pred = str(df.loc[row_index, "Predicción"])
    hit = int(result == pred)

    df.loc[row_index, "Evaluada"] = 1
    df.loc[row_index, "Fecha evaluación"] = eval_date
    df.loc[row_index, "Precio cierre"] = close_price
    df.loc[row_index, "Variación %"] = variation * 100
    df.loc[row_index, "Resultado"] = result
    df.loc[row_index, "Acierto"] = hit
    save_predictions(df)

    message = f"Predijo {pred}. Resultado real {result}. {'Acertó' if hit else 'Falló'}."
    return True, message


def busa_ai_confidence_label(prob_dict):
    max_prob = max(float(prob_dict.get("Sube", 0)), float(prob_dict.get("Baja", 0)), float(prob_dict.get("Lateral", 0)))
    if max_prob >= 0.70:
        return "Muy alta"
    if max_prob >= 0.60:
        return "Alta"
    if max_prob >= 0.52:
        return "Media"
    return "Baja"

def busa_ai_recommended_strategy(prediction, confidence):
    if prediction == "Sube":
        return "Call comprado" if confidence in ["Alta", "Muy alta"] else "Bull Call Spread"
    if prediction == "Baja":
        return "Put comprado" if confidence in ["Alta", "Muy alta"] else "Bear Put Spread"
    return "Estrategia lateral / esperar"

def option_strategy_suggestions(prediction, confidence):
    if prediction == "Sube":
        return [
            {"Estrategia": "Call comprado", "Escenario": "Alcista fuerte", "Ganancia": "Ilimitada teórica", "Pérdida máxima": "Prima pagada", "Comentario": "Mayor potencial, pero alto theta si no sube rápido."},
            {"Estrategia": "Bull Call Spread", "Escenario": "Alcista moderado", "Ganancia": "Limitada", "Pérdida máxima": "Costo neto", "Comentario": "Reduce costo y riesgo, limita ganancia."},
        ]
    if prediction == "Baja":
        return [
            {"Estrategia": "Put comprado", "Escenario": "Bajista fuerte", "Ganancia": "Alta, limitada por subyacente a cero", "Pérdida máxima": "Prima pagada", "Comentario": "Mayor potencial bajista, pero pierde por theta."},
            {"Estrategia": "Bear Put Spread", "Escenario": "Bajista moderado", "Ganancia": "Limitada", "Pérdida máxima": "Costo neto", "Comentario": "Reduce costo y riesgo, limita ganancia."},
        ]
    return [
        {"Estrategia": "No operar / esperar", "Escenario": "Lateral o baja convicción", "Ganancia": "No aplica", "Pérdida máxima": "0", "Comentario": "Preservar capital también es estrategia."},
        {"Estrategia": "Straddle / Strangle comprado", "Escenario": "Movimiento fuerte sin dirección clara", "Ganancia": "Alta si se mueve fuerte", "Pérdida máxima": "Primas pagadas", "Comentario": "Necesita movimiento grande para compensar costo."},
    ]

def busa_ai_reason_cards(prob_dict, hv, S, hist_df):
    feats = technical_features(hist_df)
    reasons = []
    rsi = feats.get("RSI14", np.nan)
    ret5 = feats.get("Retorno 5d %", np.nan)
    dist20 = feats.get("Dist EMA20 %", np.nan)
    atr = feats.get("ATR14 %", np.nan)

    if not pd.isna(rsi):
        if rsi < 35:
            reasons.append("RSI bajo: posible rebote técnico")
        elif rsi > 70:
            reasons.append("RSI alto: suba extendida / posible agotamiento")
        else:
            reasons.append("RSI neutral")

    if not pd.isna(ret5):
        if ret5 > 3:
            reasons.append("Momentum 5 ruedas positivo")
        elif ret5 < -3:
            reasons.append("Momentum 5 ruedas negativo")

    if not pd.isna(dist20):
        if dist20 > 0:
            reasons.append("Precio sobre EMA20")
        else:
            reasons.append("Precio bajo EMA20")

    if not pd.isna(atr):
        if atr > 4:
            reasons.append("ATR alto: movimiento esperado amplio")
        else:
            reasons.append("ATR moderado")

    if hv > 0.40:
        reasons.append("Volatilidad histórica elevada")
    else:
        reasons.append("Volatilidad histórica controlada")

    return reasons

def busa_ai_accuracy_summary(activo):
    df = load_predictions()
    if df.empty or "Activo" not in df.columns:
        return np.nan, 0, pd.DataFrame()
    d = df[(df["Activo"].astype(str) == activo) & (pd.to_numeric(df.get("Evaluada", 0), errors="coerce").fillna(0).astype(int) == 1)].copy()
    if d.empty:
        return np.nan, 0, pd.DataFrame()
    acc = pd.to_numeric(d["Acierto"], errors="coerce").mean()
    return acc, len(d), d


def auto_save_daily_prediction(activo, S, prob, hv, hist_df):
    """
    Guarda una predicción por activo y por día. Evita duplicar si ya existe.
    Se ejecuta cuando la app tiene datos actualizados.
    """
    df = load_predictions()
    today = datetime.now().strftime("%Y-%m-%d")
    if not df.empty and "Fecha señal" in df.columns:
        exists = ((df["Fecha señal"].astype(str) == today) & (df["Activo"].astype(str) == activo)).any()
        if exists:
            return False

    feats = technical_features(hist_df)
    pred = dominant_prediction(prob)

    new_row = {
        "Fecha señal": today,
        "Activo": activo,
        "Precio inicial": S,
        "Prob. suba": prob["Sube"] * 100,
        "Prob. baja": prob["Baja"] * 100,
        "Prob. lateral": prob["Lateral"] * 100,
        "Predicción": pred,
        "VH %": hv * 100,
        "RSI14": feats.get("RSI14", np.nan),
        "Retorno 1d %": feats.get("Retorno 1d %", np.nan),
        "Retorno 5d %": feats.get("Retorno 5d %", np.nan),
        "Dist EMA20 %": feats.get("Dist EMA20 %", np.nan),
        "Dist EMA50 %": feats.get("Dist EMA50 %", np.nan),
        "ATR14 %": feats.get("ATR14 %", np.nan),
        "Volumen relativo": feats.get("Volumen relativo", np.nan),
        "Evaluada": 0,
        "Fecha evaluación": "",
        "Precio cierre": np.nan,
        "Variación %": np.nan,
        "Resultado": "",
        "Acierto": np.nan,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_predictions(df)
    return True

def evaluate_pending_predictions(activo, current_price, lateral_threshold):
    """
    Evalúa predicciones previas no evaluadas.
    No evalúa las del mismo día.
    """
    df = load_predictions()
    if df.empty or "Evaluada" not in df.columns:
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    count = 0

    for idx, row in df.iterrows():
        if str(row.get("Activo")) != activo:
            continue
        if str(row.get("Fecha señal")) >= today:
            continue
        try:
            evaluated = int(float(row.get("Evaluada", 0)))
        except Exception:
            evaluated = 0
        if evaluated == 1:
            continue

        initial = clean_num(row.get("Precio inicial"))
        if np.isnan(initial) or initial <= 0:
            continue

        variation = (current_price / initial - 1)
        result = "Sube" if variation > lateral_threshold else "Baja" if variation < -lateral_threshold else "Lateral"
        pred = str(row.get("Predicción"))
        hit = int(result == pred)

        df.loc[idx, "Evaluada"] = 1
        df.loc[idx, "Fecha evaluación"] = today
        df.loc[idx, "Precio cierre"] = current_price
        df.loc[idx, "Variación %"] = variation * 100
        df.loc[idx, "Resultado"] = result
        df.loc[idx, "Acierto"] = hit
        count += 1

    if count:
        save_predictions(df)
    return count



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


def extract_iol_quote_price(raw_quote):
    """
    Extrae precio del subyacente desde IOL.
    Prioriza último precio, luego promedio/cierre anterior.
    """
    if not isinstance(raw_quote, dict):
        return np.nan

    candidates = [
        raw_quote.get("ultimoPrecio"),
        raw_quote.get("ultimo"),
        raw_quote.get("precio"),
        raw_quote.get("precioPromedio"),
        raw_quote.get("cierreAnterior"),
    ]

    cot = raw_quote.get("cotizacion")
    if isinstance(cot, dict):
        candidates.extend([
            cot.get("ultimoPrecio"),
            cot.get("ultimo"),
            cot.get("precio"),
            cot.get("precioPromedio"),
            cot.get("cierreAnterior"),
        ])

    for x in candidates:
        val = clean_num(x)
        if not np.isnan(val) and val > 0:
            return val
    return np.nan


def extract_option_quote_fields(raw_quote):
    """
    Extrae Compra, Venta, Último y Volumen desde cotización individual IOL.
    Está preparado para estructuras con cotizacion/puntas en dict o lista.
    """
    if not isinstance(raw_quote, dict):
        return {}

    cot = raw_quote.get("cotizacion") if isinstance(raw_quote.get("cotizacion"), dict) else raw_quote

    puntas_raw = cot.get("puntas") or raw_quote.get("puntas")
    if isinstance(puntas_raw, list) and len(puntas_raw) > 0 and isinstance(puntas_raw[0], dict):
        puntas = puntas_raw[0]
    elif isinstance(puntas_raw, dict):
        puntas = puntas_raw
    else:
        puntas = {}

    compra = clean_num(
        puntas.get("precioCompra")
        or puntas.get("compra")
        or cot.get("precioCompra")
        or cot.get("compra")
        or raw_quote.get("precioCompra")
        or raw_quote.get("compra")
    )

    venta = clean_num(
        puntas.get("precioVenta")
        or puntas.get("venta")
        or cot.get("precioVenta")
        or cot.get("venta")
        or raw_quote.get("precioVenta")
        or raw_quote.get("venta")
    )

    ultimo = clean_num(
        cot.get("ultimoPrecio")
        or cot.get("ultimo")
        or cot.get("precio")
        or raw_quote.get("ultimoPrecio")
        or raw_quote.get("ultimo")
        or raw_quote.get("precio")
    )

    volumen = clean_num(
        cot.get("volumen")
        or cot.get("volumenNominal")
        or raw_quote.get("volumen")
        or raw_quote.get("volumenNominal")
    )

    return {
        "Compra": compra,
        "Venta": venta,
        "Último": ultimo,
        "Volumen": volumen,
    }

def merge_top_quotes_into_options(options_df, quotes_by_ticker):
    """
    Actualiza solo las filas consultadas individualmente.
    Si un campo viene vacío desde IOL, conserva el valor anterior.
    """
    if options_df is None or options_df.empty:
        return options_df

    df = options_df.copy()
    for ticker, fields in quotes_by_ticker.items():
        mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
        if not mask.any():
            continue
        for col in ["Compra", "Venta", "Último", "Volumen"]:
            val = fields.get(col, np.nan)
            if not np.isnan(clean_num(val)):
                df.loc[mask, col] = val
    return df

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

        puntas_raw = cot.get("puntas") or item.get("puntas")
        if isinstance(puntas_raw, list) and len(puntas_raw) > 0 and isinstance(puntas_raw[0], dict):
            puntas = puntas_raw[0]
        elif isinstance(puntas_raw, dict):
            puntas = puntas_raw
        else:
            puntas = {}

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
            "Compra": clean_num(
                puntas.get("precioCompra")
                or puntas.get("compra")
                or cot.get("precioCompra")
                or cot.get("compra")
                or item.get("precioCompra")
                or item.get("compra")
            ),
            "Venta": clean_num(
                puntas.get("precioVenta")
                or puntas.get("venta")
                or cot.get("precioVenta")
                or cot.get("venta")
                or item.get("precioVenta")
                or item.get("venta")
            ),
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
# Estrategias
# =========================
STRATEGIES = {
    "Call comprado": "Alcista fuerte. Riesgo limitado a la prima pagada.",
    "Put comprado": "Bajista. Riesgo limitado a la prima pagada.",
    "Bull Call Spread": "Alcista moderada. Compra call baja y vende call más alta.",
    "Bear Put Spread": "Bajista moderada. Compra put alta y vende put más baja.",
    "Straddle comprado": "Apuesta a fuerte movimiento en cualquier dirección.",
    "Strangle comprado": "Apuesta a movimiento fuerte con menor costo que straddle.",
}

def leg_payoff(price, typ, K, premium, side, qty=1):
    if typ == "call":
        val = np.maximum(price - K, 0) - premium
    else:
        val = np.maximum(K - price, 0) - premium
    if side == "sell":
        val = -val
    return val * qty

def strategy_metrics(payoff, net_cost):
    max_gain = float(np.nanmax(payoff))
    max_loss = float(np.nanmin(payoff))
    risk_capital = abs(float(net_cost)) if abs(float(net_cost)) > 0 else abs(max_loss)
    loss_pct = abs(max_loss) / risk_capital * 100 if risk_capital and risk_capital > 0 else np.nan
    unlimited_upside = payoff[-1] > payoff[-2] and payoff[-1] > payoff[len(payoff)//2]
    return {"max_gain": max_gain, "max_loss": max_loss, "risk_capital": risk_capital, "loss_pct": loss_pct, "unlimited_upside": unlimited_upside}

def first_valid_price(row):
    for col in ["Prima usada", "Último", "Venta", "Compra"]:
        val = clean_num(row.get(col))
        if not np.isnan(val) and val > 0:
            return float(val)
    return np.nan

def estimate_strategy_success(strategy_type, breakeven, S, prob_dict):
    """
    Estimación educativa simple:
    usa la dirección dominante + distancia al break-even.
    Luego se podrá reemplazar por modelo entrenado.
    """
    if S <= 0 or breakeven is None or np.isnan(breakeven):
        return np.nan

    if strategy_type in ["Call comprado", "Bull Call Spread"]:
        base = float(prob_dict.get("Sube", 0))
        distance_penalty = max(0, (breakeven / S - 1)) * 2.0
        return max(0, min(1, base - distance_penalty))

    if strategy_type in ["Put comprado", "Bear Put Spread"]:
        base = float(prob_dict.get("Baja", 0))
        distance_penalty = max(0, (1 - breakeven / S)) * 2.0
        return max(0, min(1, base - distance_penalty))

    return float(prob_dict.get("Lateral", 0))


# =========================
# Motor de payoff / estrategias
# =========================
def leg_payoff(price, typ, K, premium, side, qty=1):
    if typ == "call":
        val = np.maximum(price - K, 0) - premium
    else:
        val = np.maximum(K - price, 0) - premium
    if side == "sell":
        val = -val
    return val * qty

def strategy_payoff(legs, prices):
    total = np.zeros_like(prices, dtype=float)
    net_cost = 0.0
    for side, typ, K, premium, qty in legs:
        total += leg_payoff(prices, typ, K, premium, side, qty)
        net_cost += premium * qty * (1 if side == "buy" else -1)
    return total, net_cost

def strategy_metrics(payoff, net_cost):
    max_gain = float(np.nanmax(payoff))
    max_loss = float(np.nanmin(payoff))
    risk_capital = abs(float(net_cost)) if abs(float(net_cost)) > 0 else abs(max_loss)
    loss_pct = abs(max_loss) / risk_capital * 100 if risk_capital and risk_capital > 0 else np.nan
    unlimited_upside = payoff[-1] > payoff[-2] and payoff[-1] > payoff[len(payoff)//2]
    return {
        "max_gain": max_gain,
        "max_loss": max_loss,
        "risk_capital": risk_capital,
        "loss_pct": loss_pct,
        "unlimited_upside": unlimited_upside,
    }

def build_strategy_advisor(analyzed, S, prob_dict, max_loss_pct_limit=100):
    """
    Strategy Advisor 9.1.
    Evalúa estrategias educativas con opciones disponibles:
    - Call comprado
    - Put comprado
    - Bull Call Spread
    - Bear Put Spread
    - Long Straddle
    - Long Strangle
    - Long Call Butterfly
    - Long Iron Condor / expectativa lateral
    """
    if analyzed is None or analyzed.empty:
        return pd.DataFrame()

    df = analyzed.copy()
    df["prima_ref"] = df.apply(first_valid_price, axis=1)
    df = df.dropna(subset=["prima_ref", "Strike", "Score Busa"])
    df = df[df["prima_ref"] > 0]
    if df.empty:
        return pd.DataFrame()

    rows = []
    prices = np.linspace(S * 0.65, S * 1.40, 350)
    pred = dominant_prediction(prob_dict)

    calls = df[df["Tipo"] == "CALL"].sort_values("Strike")
    puts = df[df["Tipo"] == "PUT"].sort_values("Strike")

    def add_row(strategy, t1, t2, t3, t4, escenario, ganancia, legs, breakeven, success, score_base, comment):
        payoff, net_cost = strategy_payoff(legs, prices)
        m = strategy_metrics(payoff, net_cost)
        roi = abs(m["max_gain"] / abs(m["max_loss"])) if m["max_loss"] else np.nan
        strategy_score = (
            score_base * 0.45 +
            (success * 100 if not np.isnan(success) else 45) * 0.35 +
            min(20, roi * 8 if not np.isnan(roi) else 0)
        )
        if m["unlimited_upside"]:
            strategy_score += 12
        rows.append({
            "Ranking": "",
            "Estrategia": strategy,
            "Ticker 1": t1,
            "Ticker 2": t2,
            "Ticker 3": t3,
            "Ticker 4": t4,
            "Escenario": escenario,
            "Ganancia": ganancia,
            "Costo neto": net_cost,
            "Pérdida máx.": m["max_loss"],
            "% pérdida/capital": m["loss_pct"],
            "Break-even": breakeven,
            "Prob. éxito est. %": success * 100 if not np.isnan(success) else np.nan,
            "Score estrategia": strategy_score,
            "Comentario": comment,
        })

    # Long calls: upside ilimitado
    if pred == "Sube" and not calls.empty:
        candidate_calls = calls[(calls["Strike"] >= S * 0.92) & (calls["Strike"] <= S * 1.18)].sort_values("Score Busa", ascending=False).head(10)
        for _, c in candidate_calls.iterrows():
            K = float(c["Strike"]); p = float(c["prima_ref"])
            legs = [("buy", "call", K, p, 1)]
            breakeven = K + p
            success = estimate_strategy_success("Call comprado", breakeven, S, prob_dict)
            add_row("Call comprado", c["Ticker"], "", "", "", "Alcista fuerte", "Ilimitada teórica", legs, breakeven, success, float(c["Score Busa"]), "Mayor potencial alcista. Riesgo limitado a prima.")

    # Long puts
    if pred == "Baja" and not puts.empty:
        candidate_puts = puts[(puts["Strike"] >= S * 0.82) & (puts["Strike"] <= S * 1.08)].sort_values("Score Busa", ascending=False).head(10)
        for _, p_row in candidate_puts.iterrows():
            K = float(p_row["Strike"]); p = float(p_row["prima_ref"])
            legs = [("buy", "put", K, p, 1)]
            breakeven = K - p
            success = estimate_strategy_success("Put comprado", breakeven, S, prob_dict)
            add_row("Put comprado", p_row["Ticker"], "", "", "", "Bajista fuerte", "Alta, limitada por subyacente a cero", legs, breakeven, success, float(p_row["Score Busa"]), "Potencial bajista con riesgo limitado a prima.")

    # Bull call spreads
    if pred == "Sube" and len(calls) >= 2:
        base_calls = calls[(calls["Strike"] >= S * 0.92) & (calls["Strike"] <= S * 1.10)].sort_values("Score Busa", ascending=False).head(6)
        for _, buy in base_calls.iterrows():
            higher = calls[calls["Strike"] > buy["Strike"]].head(5)
            for _, sell in higher.iterrows():
                p_buy = float(buy["prima_ref"]); p_sell = float(sell["prima_ref"]); net = p_buy - p_sell
                if net <= 0: continue
                legs = [("buy", "call", float(buy["Strike"]), p_buy, 1), ("sell", "call", float(sell["Strike"]), p_sell, 1)]
                breakeven = float(buy["Strike"]) + net
                success = estimate_strategy_success("Bull Call Spread", breakeven, S, prob_dict)
                score_base = np.nanmean([buy["Score Busa"], sell["Score Busa"]])
                add_row("Bull Call Spread", buy["Ticker"], sell["Ticker"], "", "", "Alcista moderado", "Limitada", legs, breakeven, success, score_base, "Menor costo y menor riesgo que call comprado.")

    # Bear put spreads
    if pred == "Baja" and len(puts) >= 2:
        base_puts = puts[(puts["Strike"] >= S * 0.90) & (puts["Strike"] <= S * 1.08)].sort_values("Score Busa", ascending=False).head(6)
        for _, buy in base_puts.iterrows():
            lower = puts[puts["Strike"] < buy["Strike"]].tail(5)
            for _, sell in lower.iterrows():
                p_buy = float(buy["prima_ref"]); p_sell = float(sell["prima_ref"]); net = p_buy - p_sell
                if net <= 0: continue
                legs = [("buy", "put", float(buy["Strike"]), p_buy, 1), ("sell", "put", float(sell["Strike"]), p_sell, 1)]
                breakeven = float(buy["Strike"]) - net
                success = estimate_strategy_success("Bear Put Spread", breakeven, S, prob_dict)
                score_base = np.nanmean([buy["Score Busa"], sell["Score Busa"]])
                add_row("Bear Put Spread", buy["Ticker"], sell["Ticker"], "", "", "Bajista moderado", "Limitada", legs, breakeven, success, score_base, "Menor costo y menor riesgo que put comprado.")

    # Straddle / Strangle long for movement
    if len(calls) >= 1 and len(puts) >= 1:
        near_calls = calls.iloc[(calls["Strike"] - S).abs().argsort()[:4]]
        near_puts = puts.iloc[(puts["Strike"] - S).abs().argsort()[:4]]
        for _, c in near_calls.iterrows():
            for _, p_row in near_puts.iterrows():
                pc = float(c["prima_ref"]); pp = float(p_row["prima_ref"])
                legs = [("buy", "call", float(c["Strike"]), pc, 1), ("buy", "put", float(p_row["Strike"]), pp, 1)]
                total_p = pc + pp
                if abs(float(c["Strike"]) - float(p_row["Strike"])) < 1e-9:
                    strat = "Long Straddle"
                    bkeven = np.nan
                else:
                    strat = "Long Strangle"
                    bkeven = np.nan
                score_base = np.nanmean([c["Score Busa"], p_row["Score Busa"]])
                success = max(float(prob_dict.get("Sube", 0)), float(prob_dict.get("Baja", 0))) if pred != "Lateral" else np.nan
                add_row(strat, c["Ticker"], p_row["Ticker"], "", "", "Movimiento fuerte", "Ilimitada al alza / alta a la baja", legs, bkeven, success, score_base, "Apuesta a movimiento fuerte. Riesgo limitado a primas.")

    # Butterfly with calls: lateral / target
    if len(calls) >= 3:
        strikes = sorted(calls["Strike"].unique())
        for i in range(1, len(strikes)-1):
            k1, k2, k3 = strikes[i-1], strikes[i], strikes[i+1]
            if abs((k2-k1) - (k3-k2)) > 1e-6:
                continue
            if not (S*0.90 <= k2 <= S*1.10):
                continue
            r1 = calls[calls["Strike"] == k1].sort_values("Score Busa", ascending=False).head(1)
            r2 = calls[calls["Strike"] == k2].sort_values("Score Busa", ascending=False).head(1)
            r3 = calls[calls["Strike"] == k3].sort_values("Score Busa", ascending=False).head(1)
            if r1.empty or r2.empty or r3.empty:
                continue
            r1, r2, r3 = r1.iloc[0], r2.iloc[0], r3.iloc[0]
            legs = [
                ("buy", "call", float(k1), float(r1["prima_ref"]), 1),
                ("sell", "call", float(k2), float(r2["prima_ref"]), 2),
                ("buy", "call", float(k3), float(r3["prima_ref"]), 1),
            ]
            score_base = np.nanmean([r1["Score Busa"], r2["Score Busa"], r3["Score Busa"]])
            add_row("Long Call Butterfly", r1["Ticker"], r2["Ticker"], r3["Ticker"], "", "Lateral / objetivo cercano", "Limitada", legs, np.nan, float(prob_dict.get("Lateral", 0)), score_base, "Riesgo definido. Busca cierre cerca del strike central.")
            break

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("Score estrategia", ascending=False).reset_index(drop=True)
    out["Ranking"] = np.arange(1, len(out)+1)
    return out

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
            client = IOLClient.from_config()

            raw = client.get_options(TICKERS[activo]["iol"])
            quote_raw = client.get_quote(TICKERS[activo]["iol"])
            spot_iol = extract_iol_quote_price(quote_raw)

            st.session_state["raw_iol"] = raw
            st.session_state["quote_iol"] = quote_raw
            st.session_state["options_df"] = normalize_options(raw)

            if not np.isnan(spot_iol):
                st.session_state["spot_iol"] = float(spot_iol)

            st.session_state["last_update"] = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")

            save_snapshot(
                activo,
                raw,
                spot=float(spot_iol) if not np.isnan(spot_iol) else None,
                quote_raw=quote_raw,
            )

            add_calls(3)
            st.success("Mercado actualizado: opciones + subyacente IOL + snapshot. Consumo estimado: 3 consultas.")
            st.rerun()

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
                if snap.get("quote_raw") is not None:
                    st.session_state["quote_iol"] = snap.get("quote_raw")
                if snap.get("spot") is not None:
                    st.session_state["spot_iol"] = float(snap.get("spot"))
                st.session_state["last_update"] = snap.get("saved_at")
                st.success(f"Snapshot cargado: {snap.get('saved_at')}")
            else:
                st.warning("No hay datos cargados. Primero tocá Actualizar mercado.")
        else:
            st.info("Recalculando con datos ya cargados. No consume API IOL.")
        st.rerun()


    st.divider()
    top_n_quotes = st.number_input("TOP puntas a consultar", min_value=1, max_value=20, value=10, step=1)
    if st.button("🔎 Traer puntas opciones TOP"):
        if "options_df" not in st.session_state or st.session_state.get("options_df", pd.DataFrame()).empty:
            st.warning("Primero cargá opciones con Actualizar mercado.")
        else:
            try:
                client = IOLClient.from_config()

                # Calcula ranking actual para decidir TOP sin depender de la tabla visible
                h_tmp = get_hist(TICKERS[activo]["local"], period)
                close_tmp = h_tmp["Close"].dropna()
                s_tmp = float(st.session_state.get("spot_iol", float(close_tmp.iloc[-1])))
                prob_tmp = prob_data(close_tmp, int(horizon), lateral, int(lookback))
                hv_tmp = prob_tmp["VH"]
                t_tmp = days / 365

                ranked = analyze(
                    st.session_state["options_df"],
                    s_tmp,
                    t_tmp,
                    r,
                    hv_tmp,
                    prob_tmp["Sube"],
                    prob_tmp["Baja"],
                    mode,
                )

                top = (
                    ranked.dropna(subset=["Score Busa"])
                    .sort_values("Score Busa", ascending=False)
                    .head(int(top_n_quotes))
                )

                tickers_top = top["Ticker"].dropna().astype(str).str.upper().unique().tolist()

                quotes = {}
                raw_quotes = {}
                for tk in tickers_top:
                    q = client.get_quote(tk)
                    raw_quotes[tk] = q
                    quotes[tk] = extract_option_quote_fields(q)

                st.session_state["top_quotes_raw"] = raw_quotes
                st.session_state["options_df"] = merge_top_quotes_into_options(
                    st.session_state["options_df"],
                    quotes,
                )

                add_calls(len(tickers_top))
                st.success(f"Puntas TOP actualizadas: {len(tickers_top)} opciones. Consumo estimado: {len(tickers_top)} consultas.")
                st.rerun()

            except (FileNotFoundError, IOLAuthError, IOLApiError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error actualizando puntas TOP: {e}")

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
S_yf = float(close.iloc[-1])
S = float(st.session_state.get("spot_iol", S_yf))
prob = prob_data(close, int(horizon), lateral, int(lookback))
prob = apply_learning_to_probabilities(prob, activo)
hv = prob["VH"]
T = days / 365

if "options_df" not in st.session_state or st.session_state.get("options_df", pd.DataFrame()).empty:
    snap = load_snapshot(activo)
    if snap:
        st.session_state["raw_iol"] = snap["raw"]
        st.session_state["options_df"] = normalize_options(snap["raw"])
        if snap.get("quote_raw") is not None:
            st.session_state["quote_iol"] = snap.get("quote_raw")
        if snap.get("spot") is not None:
            st.session_state["spot_iol"] = float(snap.get("spot"))
        st.session_state["last_update"] = snap.get("saved_at")

df_options = st.session_state.get("options_df", pd.DataFrame())
analyzed = pd.DataFrame()
if not df_options.empty:
    analyzed = analyze(df_options, S, T, r, hv, prob["Sube"], prob["Baja"], mode)

# =========================
# UI
# =========================
tabs = st.tabs(["Dashboard", "Opciones", "Probabilidades", "Busa AI", "Advisor", "Estrategias", "Velas", "Favoritos"])

with tabs[0]:
    st.subheader(f"Dashboard {activo}")
    c1, c2 = st.columns(2)
    c1.metric("Precio", f"{S:,.2f}")
    st.caption("Fuente precio: IOL" if "spot_iol" in st.session_state else "Fuente precio: yfinance")
    c2.metric("VH", f"{hv*100:.1f}%")
    c3, c4 = st.columns(2)
    c3.metric("Prob. suba", f"{prob['Sube']:.1%}")
    st.caption(f"Learning factor: {prob.get('Learning factor', 1.0):.2f} | Señales evaluadas: {prob.get('Learning n', 0)}")
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
    st.subheader("Busa AI")
    st.caption("Centro de inteligencia: señal actual, aprendizaje histórico, evaluación automática y explicación del modelo.")

    pred = dominant_prediction(prob)
    confidence = busa_ai_confidence_label(prob)
    strategy = busa_ai_recommended_strategy(pred, confidence)
    acc, n_eval, d_eval = busa_ai_accuracy_summary(activo)

    st.markdown("### Señal actual")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activo", activo)
    c2.metric("Predicción", pred)
    c3.metric("Confianza", confidence)
    c4.metric("Estrategia sugerida", strategy)

    st.markdown("### Probabilidades")
    p1, p2, p3 = st.columns(3)
    p1.metric("Sube", f"{prob['Sube']:.1%}")
    p2.metric("Baja", f"{prob['Baja']:.1%}")
    p3.metric("Lateral", f"{prob['Lateral']:.1%}")

    st.markdown("### Aprendizaje visible")
    st.caption("Evaluada=0 significa pendiente. Cuando se evalúa, Resultado y Acierto muestran si predijo bien o mal.")
    l1, l2, l3 = st.columns(3)
    l1.metric("Learning factor", f"{prob.get('Learning factor', 1.0):.2f}")
    l2.metric("Señales evaluadas", n_eval)
    l3.metric("Accuracy histórico", "" if pd.isna(acc) else f"{acc*100:.1f}%")

    if prob.get("Learning factor", 1.0) == 1.0:
        st.info("El modelo todavía está neutral: necesita más señales evaluadas o el accuracy no justifica ajustar.")
    elif prob.get("Learning factor", 1.0) > 1.0:
        st.success("El modelo está reforzando la predicción dominante porque el historial viene acompañando.")
    else:
        st.warning("El modelo está moderando la predicción dominante porque el historial viene fallando.")

    st.markdown("### Estrategias sugeridas por Busa AI")
    st.dataframe(pd.DataFrame(option_strategy_suggestions(pred, confidence)), use_container_width=True)

    st.markdown("### Por qué Busa AI interpreta esto")
    reasons = busa_ai_reason_cards(prob, hv, S, h)
    for r_reason in reasons[:6]:
        st.write(f"✔ {r_reason}")

    st.markdown("### Base vs ajustada por Learning")
    st.dataframe(pd.DataFrame([
        {"Escenario": "Sube", "Base %": prob.get("Sube base", prob["Sube"])*100, "Ajustada %": prob["Sube"]*100},
        {"Escenario": "Baja", "Base %": prob.get("Baja base", prob["Baja"])*100, "Ajustada %": prob["Baja"]*100},
        {"Escenario": "Lateral", "Base %": prob.get("Lateral base", prob["Lateral"])*100, "Ajustada %": prob["Lateral"]*100},
    ]), use_container_width=True)

    st.markdown("### Acciones del motor")
    col_a, col_b = st.columns(2)
    if col_a.button("💾 Guardar señal diaria ahora"):
        created = auto_save_daily_prediction(activo, S, prob, hv, h)
        if created:
            st.success("Señal diaria guardada.")
        else:
            st.info("Ya existía una señal para este activo en la fecha de hoy.")
        st.rerun()

    if col_b.button("✅ Evaluar automáticamente pendientes"):
        count = evaluate_pending_predictions_auto(activo, h, lateral)
        if count:
            st.success(f"Se evaluaron automáticamente {count} señales pendientes.")
        else:
            st.info("No había señales pendientes con cierre posterior disponible.")
        st.rerun()

    df_pred = load_predictions()
    if not df_pred.empty and "Activo" in df_pred.columns:
        d = df_pred[df_pred["Activo"] == activo].copy()
        if not d.empty:
            st.markdown("### Señales guardadas")
            st.dataframe(add_learning_status_columns(d).tail(50), use_container_width=True)

            if not d_eval.empty:
                by_pred = d_eval.groupby("Predicción")["Acierto"].mean().reset_index()
                by_pred["Acierto"] = by_pred["Acierto"] * 100
                st.markdown("### Acierto por tipo de predicción")
                st.dataframe(by_pred, use_container_width=True)
        else:
            st.info("Todavía no hay señales para este activo.")
    else:
        st.info("Todavía no hay señales guardadas.")

    with st.expander("Carga manual de respaldo", expanded=False):
        st.caption("Usalo solo si querés corregir o cargar manualmente un cierre puntual.")
        close_price = st.number_input("Precio cierre / resultado real", min_value=0.0, value=float(S), step=1.0)
        lateral_threshold = st.number_input(
            "Umbral lateral +/- %",
            min_value=0.1,
            max_value=20.0,
            value=float(lateral * 100),
            step=0.1,
        ) / 100

        variation = (close_price / S - 1) if S else 0.0
        result = "Sube" if variation > lateral_threshold else "Baja" if variation < -lateral_threshold else "Lateral"
        hit = int(result == pred)
        st.write(f"Resultado calculado: **{result}** | Variación: **{variation*100:.2f}%** | Acierto: **{'Sí' if hit else 'No'}**")

        if st.button("Registrar resultado manual"):
            df_learn = load_learning()
            new_row = {
                "Fecha": datetime.now().strftime("%Y-%m-%d"),
                "Activo": activo,
                "Precio inicial": S,
                "Precio cierre": close_price,
                "Variación %": variation * 100,
                "Prob. suba": prob["Sube"] * 100,
                "Prob. baja": prob["Baja"] * 100,
                "Prob. lateral": prob["Lateral"] * 100,
                "Predicción": pred,
                "Resultado": result,
                "Acierto": hit,
            }
            df_learn = pd.concat([df_learn, pd.DataFrame([new_row])], ignore_index=True)
            save_learning(df_learn)
            st.success("Resultado manual registrado.")
            st.rerun()



with tabs[4]:
    st.subheader("Strategy Advisor 9.0")
    st.caption("Motor educativo basado en estrategias clásicas: long call/put, spreads, straddle, strangle y butterfly. Busca alternativas con opciones disponibles y rankea riesgo/beneficio.")

    if analyzed.empty:
        st.warning("Primero cargá opciones con Actualizar mercado (IOL).")
    else:
        pred_adv = dominant_prediction(prob)
        confidence_adv = busa_ai_confidence_label(prob)
        st.info(f"Escenario detectado: **{pred_adv}** | Confianza: **{confidence_adv}**")

        max_loss_pct = st.slider("Referencia máxima de pérdida sobre capital (%)", 10, 200, 100, 5)
        advisor = build_strategy_advisor(analyzed, S, prob, max_loss_pct)

        if advisor.empty:
            st.warning("No encontré estrategias suficientes con los datos actuales. Probá actualizar puntas TOP o revisar primas.")
        else:
            best = advisor.iloc[0]
            st.markdown("### Recomendación Busa")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estrategia", best["Estrategia"])
            c2.metric("Ticker principal", best["Ticker 1"])
            c3.metric("Score estrategia", f"{best['Score estrategia']:.0f}")
            c4.metric("Prob. éxito est.", "" if pd.isna(best["Prob. éxito est. %"]) else f"{best['Prob. éxito est. %']:.1f}%")

            st.write(f"**Comentario:** {best['Comentario']}")
            st.write(f"**Ganancia:** {best['Ganancia']}")
            st.write(f"**Pérdida máxima estimada:** {best['Pérdida máx.']:.2f}")
            if not pd.isna(best["% pérdida/capital"]):
                st.write(f"**% pérdida sobre capital arriesgado:** {best['% pérdida/capital']:.1f}%")
            if not pd.isna(best["Break-even"]):
                st.write(f"**Break-even:** {best['Break-even']:.2f}")

            st.markdown("### Ranking de estrategias")
            st.dataframe(advisor.head(20), use_container_width=True)

            # Payoff de la estrategia ganadora
            prices_adv = np.linspace(S * 0.70, S * 1.35, 300)
            legs_adv = []
            if best["Estrategia"] == "Call comprado":
                row = analyzed[analyzed["Ticker"] == best["Ticker 1"]].iloc[0]
                legs_adv = [("buy", "call", float(row["Strike"]), float(first_valid_price(row)), 1)]
            elif best["Estrategia"] == "Put comprado":
                row = analyzed[analyzed["Ticker"] == best["Ticker 1"]].iloc[0]
                legs_adv = [("buy", "put", float(row["Strike"]), float(first_valid_price(row)), 1)]
            elif best["Estrategia"] == "Bull Call Spread":
                r1 = analyzed[analyzed["Ticker"] == best["Ticker 1"]].iloc[0]
                r2 = analyzed[analyzed["Ticker"] == best["Ticker 2"]].iloc[0]
                legs_adv = [
                    ("buy", "call", float(r1["Strike"]), float(first_valid_price(r1)), 1),
                    ("sell", "call", float(r2["Strike"]), float(first_valid_price(r2)), 1),
                ]
            elif best["Estrategia"] == "Straddle/Strangle comprado":
                r1 = analyzed[analyzed["Ticker"] == best["Ticker 1"]].iloc[0]
                r2 = analyzed[analyzed["Ticker"] == best["Ticker 2"]].iloc[0]
                legs_adv = [
                    ("buy", "call", float(r1["Strike"]), float(first_valid_price(r1)), 1),
                    ("buy", "put", float(r2["Strike"]), float(first_valid_price(r2)), 1),
                ]

            if legs_adv:
                payoff_adv, net_cost_adv = strategy_payoff(legs_adv, prices_adv)
                fig_adv = go.Figure()
                fig_adv.add_trace(go.Scatter(x=prices_adv, y=payoff_adv, name="Payoff recomendación", mode="lines"))
                fig_adv.add_hline(y=0, line_dash="dash")
                fig_adv.add_vline(x=S, line_dash="dot", annotation_text="Precio actual")
                fig_adv.update_layout(template="plotly_dark", height=460, xaxis_title="Precio al vencimiento", yaxis_title="Resultado", hovermode="x unified")
                st.plotly_chart(fig_adv, use_container_width=True)

            st.caption("Herramienta educativa. No constituye recomendación financiera personalizada.")


with tabs[5]:
    st.subheader("Payoff manual")
    st.caption("Simulador manual de payoff. Para recomendaciones usá Advisor.")

    strategy_name = st.selectbox("Estrategia", list(STRATEGIES.keys()))
    st.info(STRATEGIES[strategy_name])

    legs = []
    base_strike = float(round(S / 100) * 100)

    if strategy_name == "Call comprado":
        K = st.number_input("Strike", value=base_strike, step=100.0)
        p = st.number_input("Prima pagada", value=100.0, step=1.0)
        legs = [("buy", "call", K, p, 1)]

    elif strategy_name == "Put comprado":
        K = st.number_input("Strike", value=base_strike, step=100.0)
        p = st.number_input("Prima pagada", value=100.0, step=1.0)
        legs = [("buy", "put", K, p, 1)]

    elif strategy_name == "Bull Call Spread":
        K1 = st.number_input("Strike call comprada", value=base_strike, step=100.0)
        p1 = st.number_input("Prima call comprada", value=100.0, step=1.0)
        K2 = st.number_input("Strike call vendida", value=base_strike + 400, step=100.0)
        p2 = st.number_input("Prima call vendida", value=50.0, step=1.0)
        legs = [("buy", "call", K1, p1, 1), ("sell", "call", K2, p2, 1)]

    elif strategy_name == "Bear Put Spread":
        K1 = st.number_input("Strike put comprada", value=base_strike, step=100.0)
        p1 = st.number_input("Prima put comprada", value=100.0, step=1.0)
        K2 = st.number_input("Strike put vendida", value=base_strike - 400, step=100.0)
        p2 = st.number_input("Prima put vendida", value=50.0, step=1.0)
        legs = [("buy", "put", K1, p1, 1), ("sell", "put", K2, p2, 1)]

    elif strategy_name == "Straddle comprado":
        K = st.number_input("Strike común", value=base_strike, step=100.0)
        pc = st.number_input("Prima call", value=100.0, step=1.0)
        pp = st.number_input("Prima put", value=100.0, step=1.0)
        legs = [("buy", "call", K, pc, 1), ("buy", "put", K, pp, 1)]

    elif strategy_name == "Strangle comprado":
        Kc = st.number_input("Strike call", value=base_strike + 300, step=100.0)
        pc = st.number_input("Prima call", value=80.0, step=1.0)
        Kp = st.number_input("Strike put", value=base_strike - 300, step=100.0)
        pp = st.number_input("Prima put", value=80.0, step=1.0)
        legs = [("buy", "call", Kc, pc, 1), ("buy", "put", Kp, pp, 1)]

    prices = np.linspace(S * 0.75, S * 1.25, 250)
    payoff, net_cost = strategy_payoff(legs, prices)

    metrics = strategy_metrics(payoff, net_cost)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Costo neto aprox.", f"{net_cost:.2f}")
    c2.metric("Ganancia", "Ilimitada*" if metrics["unlimited_upside"] else f"{metrics['max_gain']:.2f}")
    c3.metric("Pérdida máx. rango", f"{metrics['max_loss']:.2f}")
    c4.metric("% pérdida sobre capital", "" if pd.isna(metrics["loss_pct"]) else f"{metrics['loss_pct']:.1f}%")
    if metrics["unlimited_upside"]:
        st.caption("*Ilimitada teórica: el gráfico muestra solo el rango simulado.")

    signs = np.sign(payoff)
    breakevens = []
    for i in range(1, len(prices)):
        if signs[i] == 0 or signs[i] != signs[i-1]:
            breakevens.append(prices[i])
    if breakevens:
        st.caption("Break-even aprox.: " + ", ".join([f"{x:.2f}" for x in breakevens[:4]]))

    figp = go.Figure()
    figp.add_trace(go.Scatter(x=prices, y=payoff, name="Payoff", mode="lines"))
    figp.add_hline(y=0, line_dash="dash")
    figp.add_vline(x=S, line_dash="dot", annotation_text="Precio actual")
    figp.update_layout(
        template="plotly_dark",
        height=460,
        xaxis_title="Precio al vencimiento",
        yaxis_title="Resultado",
        hovermode="x unified",
    )
    st.plotly_chart(figp, use_container_width=True)


with tabs[6]:
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

with tabs[7]:
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
    if "quote_iol" in st.session_state:
        st.write("### Cotización subyacente IOL")
        st.json(st.session_state["quote_iol"])
    if "top_quotes_raw" in st.session_state:
        st.write("### Cotizaciones individuales TOP")
        st.json(st.session_state["top_quotes_raw"])
    if "raw_iol" in st.session_state:
        st.write("### Opciones IOL")
        st.json(st.session_state["raw_iol"])
    if "raw_iol" not in st.session_state and "quote_iol" not in st.session_state:
        st.info("Sin respuesta cruda.")
