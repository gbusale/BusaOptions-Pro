
import json
import hmac
import re
import calendar
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

if not hasattr(IOLClient, "get_portfolio"):
    def _iol_get_portfolio(self, pais: str = "argentina"):
        return self.get(f"/api/v2/portafolio/{pais}")
    IOLClient.get_portfolio = _iol_get_portfolio

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

st.title("BusaOptions Pro 9.19")
st.caption("IOL + Black-Scholes con vencimiento automático + Busa AI + Advisor con Ratio Backspreads + Learning bayesiano + Análisis técnico + Cartera IOL + Fundamentals ADR.")

TICKERS = {
    "GGAL": {"local": "GGAL.BA", "iol": "GGAL", "adr": "GGAL"},
    "YPF": {"local": "YPFD.BA", "iol": "YPFD", "adr": "YPF"},
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

def predictions_csv_bytes():
    df = load_predictions()
    return df.to_csv(index=False).encode("utf-8-sig")

def restore_predictions_from_upload(uploaded_file):
    if uploaded_file is None:
        return False
    df = pd.read_csv(uploaded_file)
    save_predictions(df)
    return True


def class_accuracy_stats(activo, pred_class, window=40, prior_strength=6, prior_mean=0.45):
    """
    Accuracy suavizada (Beta-Binomial) del modelo cuando predijo específicamente
    `pred_class` (Sube/Baja/Lateral) para `activo`, sobre las últimas `window`
    señales evaluadas de esa clase.

    Con pocas señales domina el prior (prior_mean) y el ajuste es casi nulo;
    a medida que se acumulan señales evaluadas, la accuracy observada pesa más.
    Esto evita que 2 o 3 aciertos/fallos seguidos muevan la probabilidad de forma
    exagerada, algo que sí le pasaba al ajuste anterior por umbrales fijos.

    Devuelve (acc_posterior, n_muestras, acc_cruda).
    """
    df = load_predictions()
    if df.empty or "Activo" not in df.columns or "Predicción" not in df.columns:
        return prior_mean, 0, np.nan

    d = df[
        (df["Activo"].astype(str) == activo)
        & (df["Predicción"].astype(str) == pred_class)
        & (pd.to_numeric(df.get("Evaluada", 0), errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if d.empty:
        return prior_mean, 0, np.nan

    d = d.tail(int(window))
    n = len(d)
    hits = float(pd.to_numeric(d["Acierto"], errors="coerce").sum())
    acc_raw = hits / n if n else np.nan

    alpha0 = prior_mean * prior_strength
    beta0 = (1 - prior_mean) * prior_strength
    acc_post = (hits + alpha0) / (n + alpha0 + beta0)

    return float(acc_post), n, (float(acc_raw) if not pd.isna(acc_raw) else np.nan)


def apply_learning_to_probabilities(prob_dict, activo, window=40, prior_strength=6, prior_mean=0.45):
    """
    Ajuste bayesiano de calibración, por clase.

    A diferencia del ajuste anterior (que sólo tocaba la probabilidad dominante
    con un factor por escalones fijos), acá cada una de las tres probabilidades
    (Sube/Baja/Lateral) se reescala según qué tan bien viene acertando Busa AI
    específicamente cuando predijo esa clase para ese activo. Después se
    renormaliza para que sigan sumando 1.

    Ejemplo: si el modelo predijo "Baja" muchas veces y acertó poco, la
    probabilidad de Baja se modera aunque "Sube" venga acertando bien — algo
    que el esquema anterior no podía distinguir porque sólo miraba la clase
    dominante de cada señal.
    """
    classes = ["Sube", "Baja", "Lateral"]
    base = {c: float(prob_dict.get(c, 0)) for c in classes}
    if sum(base.values()) <= 0:
        return prob_dict

    adjusted = {}
    stats = {}
    for c in classes:
        acc_post, n, acc_raw = class_accuracy_stats(activo, c, window, prior_strength, prior_mean)
        # factor = 1.0 cuando acc_post coincide con el prior (neutral);
        # sube hasta ~1.30 con accuracy sostenida alta, baja hasta ~0.70 con accuracy floja.
        if acc_post >= prior_mean:
            factor = 1.0 + (acc_post - prior_mean) * (0.85 / max(1e-6, 1 - prior_mean))
        else:
            factor = 1.0 - (prior_mean - acc_post) * (0.30 / max(1e-6, prior_mean))
        factor = float(np.clip(factor, 0.70, 1.30))
        adjusted[c] = base[c] * factor
        stats[c] = {"factor": factor, "n": n, "acc_raw": acc_raw, "acc_post": acc_post}

    total = sum(adjusted.values())
    if total <= 0:
        return prob_dict

    out = dict(prob_dict)
    for c in classes:
        out[f"{c} base"] = base[c]
        out[c] = adjusted[c] / total
    out["Learning stats"] = stats

    # Compat con el resto de la UI (que históricamente lee un único factor/n/acc,
    # tomado de la clase que domina el pronóstico base).
    dom = dominant_prediction(base)
    out["Learning factor"] = stats[dom]["factor"]
    out["Learning n"] = stats[dom]["n"]
    out["Learning accuracy"] = stats[dom]["acc_post"]
    return out


def forecast_quality_summary(activo, window=60):
    """
    Calidad de calibración del pronóstico sobre las últimas señales evaluadas:
    - Accuracy simple (predicción dominante vs resultado real).
    - Brier score multiclase (0 = perfecto, 2 = pésimo): compara las tres
      probabilidades guardadas en el momento de la señal contra el resultado
      real observado. A diferencia del accuracy, premia estar "bien calibrado"
      (no solo acertar la clase, sino que las probabilidades reflejen el riesgo real).
    """
    df = load_predictions()
    if df.empty or "Activo" not in df.columns:
        return None
    d = df[
        (df["Activo"].astype(str) == activo)
        & (pd.to_numeric(df.get("Evaluada", 0), errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if d.empty:
        return None
    d = d.tail(int(window))

    briers = []
    for _, row in d.iterrows():
        probs = {
            "Sube": clean_num(row.get("Prob. suba")) / 100,
            "Baja": clean_num(row.get("Prob. baja")) / 100,
            "Lateral": clean_num(row.get("Prob. lateral")) / 100,
        }
        outcome = str(row.get("Resultado"))
        if outcome not in probs or any(pd.isna(v) for v in probs.values()):
            continue
        sq_err = sum((probs[c] - (1.0 if c == outcome else 0.0)) ** 2 for c in probs)
        briers.append(sq_err)

    brier = float(np.mean(briers)) if briers else np.nan
    acc = pd.to_numeric(d["Acierto"], errors="coerce").mean()
    return {"n": len(d), "n_brier": len(briers), "brier": brier, "accuracy": acc}

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

@st.cache_data(ttl=3600 * 12, show_spinner=False)
def get_fundamentals(ticker_adr):
    """
    Datos fundamentales del ADR (la acción cotizando en el exterior, en
    USD -- NASDAQ para GGAL, NYSE para YPF). Se cachea 12 horas porque los
    fundamentales casi no cambian en el día; el botón de la UI fuerza un
    refresco manual igual (limpia el caché de esta función puntual).
    """
    try:
        info = yf.Ticker(ticker_adr).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}

def extract_fundamentals(info):
    """Extracción defensiva de los campos de yfinance que más importan para un vistazo fundamental rápido."""
    if not isinstance(info, dict) or not info:
        return {}

    def g(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None:
                return v
        return None

    return {
        "Precio ADR (USD)": g("currentPrice", "regularMarketPrice"),
        "Variación día %": g("regularMarketChangePercent"),
        "Apertura": g("regularMarketOpen", "open"),
        "Máximo día": g("dayHigh", "regularMarketDayHigh"),
        "Mínimo día": g("dayLow", "regularMarketDayLow"),
        "Máx. 52 sem.": g("fiftyTwoWeekHigh"),
        "Mín. 52 sem.": g("fiftyTwoWeekLow"),
        "Market Cap (USD)": g("marketCap"),
        "P/E (trailing)": g("trailingPE"),
        "P/E (forward)": g("forwardPE"),
        "P/B": g("priceToBook"),
        "Dividend Yield %": g("dividendYield"),
        "ROE %": g("returnOnEquity"),
        "Margen neto %": g("profitMargins"),
        "Beta": g("beta"),
        "EPS (TTM, USD)": g("trailingEps"),
        "Ingresos TTM (USD)": g("totalRevenue"),
        "Precio objetivo analistas (USD)": g("targetMeanPrice"),
        "Recomendación analistas": g("recommendationKey"),
        "Cantidad analistas": g("numberOfAnalystOpinions"),
    }

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


def sum_puntas_field(niveles, *keys):
    """
    Suma un campo (ej. cantidadCompra) a través de TODOS los niveles del
    libro de puntas que devuelva IOL, en vez de quedarse solo con el primer
    nivel (que es lo que muestra la 'caja de puntas' de un vistazo). Si IOL
    sólo manda un nivel, la suma coincide con ese único valor.
    """
    total = 0.0
    encontrado = False
    for n in niveles:
        if not isinstance(n, dict):
            continue
        for k in keys:
            v = clean_num(n.get(k))
            if not np.isnan(v):
                total += v
                encontrado = True
                break
    return total if encontrado else np.nan


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


def get_secondary_quote_byma(local_ticker, settlement="48hs"):
    """
    Segunda fuente de datos para el PAPEL (no las opciones): usa la librería
    PyOBD (datos abiertos de BYMA, sin cuenta de broker) como cruce
    independiente de IOL. Pensada para nunca romper la app: si la librería
    no está instalada, si cambió su formato de respuesta, o hay cualquier
    error de red, devuelve None sin propagar la excepción.

    Nota de honestidad: no pude probar el formato exacto de respuesta de
    PyOBD en este entorno (sandbox sin salida a internet financiero) --
    el parseo de campos en parse_byma_secondary_fields es una primera
    aproximación que puede necesitar ajuste con datos reales.
    """
    try:
        from pyobd import BymaData
    except Exception:
        return None
    try:
        client = BymaData()
        q = client.get_current_quote(local_ticker, settlement=settlement)
        if hasattr(q, "iloc"):
            try:
                q = q.iloc[0].to_dict()
            except Exception:
                q = q.to_dict() if hasattr(q, "to_dict") else None
        elif hasattr(q, "to_dict"):
            q = q.to_dict()
        return q if isinstance(q, dict) else None
    except Exception:
        return None


def parse_byma_secondary_fields(q):
    """
    Extracción best-effort de compra/venta/tamaños/volumen desde la
    respuesta de PyOBD. Prueba nombres en inglés (estilo pyhomebroker,
    con el que PyOBD dice ser compatible) y en español (estilo IOL), y
    deja en NaN lo que no encuentre -- no inventa valores.
    """
    if not isinstance(q, dict):
        return {}

    def _get(*keys):
        for k in keys:
            if k in q:
                v = clean_num(q[k])
                if not np.isnan(v):
                    return v
        return np.nan

    return {
        "Compra (BYMA)": _get("bid", "precioCompra", "compra"),
        "Venta (BYMA)": _get("ask", "precioVenta", "venta"),
        "Vol. Compra (BYMA)": _get("bid_size", "bidSize", "cantidadCompra"),
        "Vol. Venta (BYMA)": _get("ask_size", "askSize", "cantidadVenta"),
        "Volumen (BYMA)": _get("volume", "volumen", "nominalVolume", "volumenNominal"),
        "Último (BYMA)": _get("last", "ultimoPrecio", "ultimo", "close", "price"),
    }


def extract_option_quote_fields(raw_quote):
    """
    Extrae Compra, Venta, Vol. Compra, Vol. Venta, Último y Volumen desde
    cotización individual IOL. Está preparado para estructuras con
    cotizacion/puntas en dict o lista.
    """
    if not isinstance(raw_quote, dict):
        return {}

    cot = raw_quote.get("cotizacion") if isinstance(raw_quote.get("cotizacion"), dict) else raw_quote

    puntas_raw = cot.get("puntas") or raw_quote.get("puntas")
    if isinstance(puntas_raw, list) and len(puntas_raw) > 0 and isinstance(puntas_raw[0], dict):
        puntas = puntas_raw[0]
        niveles = [p for p in puntas_raw if isinstance(p, dict)]
    elif isinstance(puntas_raw, dict):
        puntas = puntas_raw
        niveles = [puntas_raw]
    else:
        puntas = {}
        niveles = []

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

    vol_compra = sum_puntas_field(niveles, "cantidadCompra")
    if np.isnan(vol_compra):
        vol_compra = clean_num(cot.get("cantidadCompra") or raw_quote.get("cantidadCompra"))

    vol_venta = sum_puntas_field(niveles, "cantidadVenta")
    if np.isnan(vol_venta):
        vol_venta = clean_num(cot.get("cantidadVenta") or raw_quote.get("cantidadVenta"))

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

    apertura = clean_num(cot.get("apertura") or raw_quote.get("apertura"))
    maximo = clean_num(cot.get("maximo") or raw_quote.get("maximo"))
    minimo = clean_num(cot.get("minimo") or raw_quote.get("minimo"))
    variacion = clean_num(
        cot.get("variacionPorcentual")
        or cot.get("variacion")
        or raw_quote.get("variacionPorcentual")
        or raw_quote.get("variacion")
    )
    cierre_anterior = clean_num(cot.get("cierreAnterior") or raw_quote.get("cierreAnterior"))
    if np.isnan(variacion) and not np.isnan(ultimo) and not np.isnan(cierre_anterior) and cierre_anterior != 0:
        variacion = (ultimo / cierre_anterior - 1) * 100

    return {
        "Compra": compra,
        "Venta": venta,
        "Vol. Compra": vol_compra,
        "Vol. Venta": vol_venta,
        "Último": ultimo,
        "Volumen": volumen,
        "Apertura": apertura,
        "Máximo": maximo,
        "Mínimo": minimo,
        "Variación %": variacion,
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
        for col in ["Compra", "Venta", "Vol. Compra", "Vol. Venta", "Último", "Volumen"]:
            if col not in df.columns:
                df[col] = np.nan
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

def _technical_tilt_score(feats):
    """
    Convierte indicadores técnicos en un sesgo de tendencia (drift) anualizado,
    acotado a +/-20 puntos porcentuales. No reemplaza al modelo estadístico:
    lo matiza levemente con contexto de corto plazo.

    - RSI extremo (sobrecompra/sobreventa): empuja levemente en sentido contrario
      (reversión), con peso moderado.
    - Momentum 5 ruedas y distancia a EMA20/EMA50: empujan a favor de la
      tendencia reciente, con peso moderado y cap individual.

    Es un heurístico transparente, no un modelo entrenado: sirve para que el
    pronóstico reaccione un poco al contexto técnico en vez de basarse
    exclusivamente en el retorno medio histórico (muy ruidoso).
    """
    score = 0.0
    rsi = feats.get("RSI14", np.nan)
    ret5 = feats.get("Retorno 5d %", np.nan)
    dist20 = feats.get("Dist EMA20 %", np.nan)
    dist50 = feats.get("Dist EMA50 %", np.nan)

    if not pd.isna(rsi):
        score += -(rsi - 50) / 50 * 0.05
    if not pd.isna(ret5):
        score += float(np.clip(ret5 / 100, -0.05, 0.05)) * 0.6
    if not pd.isna(dist20):
        score += float(np.clip(dist20 / 100, -0.05, 0.05)) * 0.4
    if not pd.isna(dist50):
        score += float(np.clip(dist50 / 100, -0.05, 0.05)) * 0.3

    return float(np.clip(score, -0.20, 0.20))


def prob_data(hist_df, horizon, lateral, lookback, drift_shrink=0.35, technical_tilt=True, tilt_strength=1.0):
    """
    Modelo de pronóstico Sube/Baja/Lateral (lognormal, tipo GBM), con dos mejoras
    respecto de la versión anterior:

    1) Drift con shrinkage: el retorno medio histórico de la ventana (`lookback`)
       es un estimador muy ruidoso de la tendencia futura -- usarlo tal cual
       (como antes) hace que rachas cortas de suba o baja se extrapolen de forma
       exagerada. Acá se lo multiplica por `drift_shrink` (0-1) y se lo acota,
       para que domine sólo parcialmente y no dispare probabilidades extremas.

    2) Volatilidad combinada: se mezcla el desvío simple de la ventana con un
       EWMA (más sensible a cambios recientes de volatilidad), en vez de usar
       sólo el desvío simple.

    Opcionalmente se suma un sesgo técnico acotado (RSI/momentum/EMAs) sobre
    el drift, activable/desactivable desde la barra lateral.
    """
    close = hist_df["Close"].dropna()
    rets = close.pct_change().dropna()
    lb = rets.tail(int(lookback))
    S = float(close.iloc[-1])

    hv_simple = float(lb.std() * np.sqrt(252))
    ewma_span = max(10, int(lookback) // 2)
    hv_ewma_series = rets.ewm(span=ewma_span, adjust=False).std()
    hv_ewma = float(hv_ewma_series.iloc[-1] * np.sqrt(252)) if not hv_ewma_series.empty else np.nan
    if pd.isna(hv_ewma) or hv_ewma <= 0:
        hv_ewma = hv_simple
    hv = float(0.4 * hv_simple + 0.6 * hv_ewma) if hv_simple > 0 else hv_ewma

    mu_hist = float(lb.mean() * 252)
    mu_cap = 0.60
    mu_shrunk = float(np.clip(mu_hist * drift_shrink, -mu_cap, mu_cap))

    feats = {}
    tilt = 0.0
    if technical_tilt:
        feats = technical_features(hist_df)
        tilt = _technical_tilt_score(feats) * tilt_strength

    mu = mu_shrunk + tilt

    T = horizon / 252
    up = S * (1 + lateral)
    down = S * (1 - lateral)
    mean_log = np.log(S) + (mu - 0.5 * hv**2) * T
    sd = hv * np.sqrt(T)
    p_down = norm.cdf((np.log(down) - mean_log) / sd)
    p_up = 1 - norm.cdf((np.log(up) - mean_log) / sd)
    p_up = float(np.clip(p_up, 0.0, 1.0))
    p_down = float(np.clip(p_down, 0.0, 1.0))
    p_lateral = max(0.0, 1 - p_up - p_down)

    return {
        "VH": hv,
        "VH simple": hv_simple,
        "VH EWMA": hv_ewma,
        "Mu hist": mu_hist,
        "Mu ajustada": mu,
        "Tilt técnico": tilt,
        "Sube": p_up,
        "Baja": p_down,
        "Lateral": p_lateral,
        "Nivel suba": up,
        "Nivel baja": down,
        "S": S,
        "T": T,
        "features": feats,
    }

def infer_tipo(symbol):
    s = str(symbol).upper()
    return "put" if ("GFGV" in s or "YPFV" in s or s.endswith("V")) else "call"


# =========================
# Vencimiento automático por ticker (BYMA: tercer viernes del mes)
# =========================
# Código de mes = últimas 2 letras del ticker de la opción (ej. "AG" en
# GFGC8600AG = Agosto). Confirmados contra fuentes públicas de BYMA/brokers:
# FE=Febrero, MY=Mayo, JU=Junio, AG=Agosto, DI=Diciembre. El resto (EN, MA,
# AB, JL, SE, OC, NO) se completó por el mismo patrón de no-colisión que
# usan los confirmados (ej. Mayo="MY" en vez de "MA" para no chocar con
# Marzo; Junio="JU" en vez de Julio para no chocar con Julio="JL").
# Si algún código no coincide en la práctica, se corrige acá en un solo lugar.
MESES_OPCIONES = {
    "EN": 1, "FE": 2, "MA": 3, "AB": 4, "MY": 5, "JU": 6,
    "JL": 7, "AG": 8, "SE": 9, "OC": 10, "NO": 11, "DI": 12,
}

def third_friday(year, month):
    """Tercer viernes del mes/año dado (regla de vencimiento de opciones BYMA)."""
    cal = calendar.Calendar(firstweekday=0)
    fridays = [d for d in cal.itermonthdates(year, month) if d.month == month and d.weekday() == 4]
    return fridays[2]

def parse_option_expiry(ticker, hoy=None):
    """
    Infiere la fecha de vencimiento de una opción de BYMA a partir del código
    de mes en las últimas 2 letras del ticker + la regla del tercer viernes.

    Devuelve (fecha, dias, fuente):
    - fecha: date del vencimiento, o None si no se pudo inferir.
    - dias: días corridos hasta el vencimiento (>=0), o None.
    - fuente: "ticker" si se pudo inferir, "desconocido" si no (quien llama
      debe usar en ese caso el valor manual de 'Días vencimiento' como
      respaldo).

    Nota: BYMA corre el vencimiento al hábil inmediato anterior si el tercer
    viernes cae feriado. Ese ajuste no se aplica acá (no tengo un calendario
    de feriados bursátiles argentinos confiable a mano), así que en esos
    casos puntuales la fecha real puede ser 1 día hábil antes de la
    calculada.
    """
    if hoy is None:
        hoy = datetime.now().date()
    m = re.search(r"([A-Z]{2})$", str(ticker).upper())
    if not m or m.group(1) not in MESES_OPCIONES:
        return None, None, "desconocido"

    mes = MESES_OPCIONES[m.group(1)]
    anio = hoy.year
    fecha = third_friday(anio, mes)
    if fecha < hoy:
        anio += 1
        fecha = third_friday(anio, mes)

    dias = (fecha - hoy).days
    return fecha, dias, "ticker"

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
            niveles = [p for p in puntas_raw if isinstance(p, dict)]
        elif isinstance(puntas_raw, dict):
            puntas = puntas_raw
            niveles = [puntas_raw]
        else:
            puntas = {}
            niveles = []

        vol_compra_item = sum_puntas_field(niveles, "cantidadCompra")
        if np.isnan(vol_compra_item):
            vol_compra_item = clean_num(cot.get("cantidadCompra") or item.get("cantidadCompra"))
        vol_venta_item = sum_puntas_field(niveles, "cantidadVenta")
        if np.isnan(vol_venta_item):
            vol_venta_item = clean_num(cot.get("cantidadVenta") or item.get("cantidadVenta"))

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
            "Vol. Compra": vol_compra_item,
            "Vol. Venta": vol_venta_item,
            "Último": clean_num(cot.get("ultimoPrecio") or cot.get("ultimo") or cot.get("precio") or item.get("ultimo")),
            "Volumen": clean_num(cot.get("volumen") or item.get("volumen")),
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["Strike"]).sort_values(["Tipo", "Strike", "Ticker"]) if not df.empty else df


def normalize_option_history(raw):
    """
    Parsea la serie histórica de UNA opción puntual (client.get_history),
    para mostrar las bandas de precio (máximo/mínimo/apertura/cierre) de esa
    opción a lo largo del tiempo -- no del subyacente.

    El schema exacto de esta respuesta de IOL no está confirmado acá (nunca
    se había conectado esta llamada a una pantalla), así que se prueban
    varios nombres de campo habituales, de forma defensiva -- igual que el
    resto de los parsers de este archivo. Si no matchea, revisar la
    respuesta cruda en el expander "Debug IOL — Histórico de opción".
    """
    if isinstance(raw, dict):
        for key in ["data", "result", "items", "cotizaciones", "serieHistorica", "titulos"]:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return pd.DataFrame()

    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fecha = item.get("fechaHora") or item.get("fecha") or item.get("fechaCotizacion")
        if fecha is None:
            continue
        apertura = clean_num(item.get("apertura") or item.get("precioApertura") or item.get("open"))
        maximo = clean_num(item.get("maximo") or item.get("precioMaximo") or item.get("max") or item.get("high"))
        minimo = clean_num(item.get("minimo") or item.get("precioMinimo") or item.get("min") or item.get("low"))
        cierre = clean_num(item.get("ultimoPrecio") or item.get("cierre") or item.get("precioCierre") or item.get("close"))
        volumen = clean_num(item.get("volumen") or item.get("volumenNominal") or item.get("volume"))
        rows.append({"fecha": fecha, "Open": apertura, "High": maximo, "Low": minimo, "Close": cierre, "Volume": volumen})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "Close"]).sort_values("fecha")

    # IOL puede devolver más de un registro por rueda (ej. snapshots
    # intradía). Se agrupa a UNA barra por día calendario con la regla OHLC
    # estándar, para que "cuántas ruedas" y las bandas de máx./mín. reflejen
    # sesiones reales, no registros sueltos.
    df["fecha_dia"] = df["fecha"].dt.normalize()
    agg = df.groupby("fecha_dia").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    agg.index.name = "fecha"
    return agg.sort_index()


def build_option_history_figure(hist_opt, ppc=None, ticker="", donchian_window=10):
    """
    Gráfico de la prima con Canal de Donchian (máximo/mínimo de las últimas
    N ruedas) + media móvil simple, en vez de Bollinger.

    Por qué Donchian y no Bollinger acá: Bollinger asume que el precio
    tiende a revertir a una media -- tiene sentido para el subyacente, pero
    NO para la prima de una opción, que tiene un decaimiento de tiempo
    (theta) sistemático hacia el vencimiento: no "revierte" a nada, se va
    achicando. El Canal de Donchian no asume reversión a la media: sólo
    marca el máximo y el mínimo efectivamente tocados en la ventana, que es
    exactamente el rango que pediste ("Min/Max de algunas ruedas").
    """
    donchian_high = hist_opt["High"].rolling(donchian_window, min_periods=1).max()
    donchian_low = hist_opt["Low"].rolling(donchian_window, min_periods=1).min()
    sma_close = hist_opt["Close"].rolling(donchian_window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_opt.index, y=donchian_high, name=f"Máx. {donchian_window}r",
        line=dict(color="rgba(139,149,165,0.55)", width=1, dash="dot"),
        hovertemplate="Máx %d ruedas: %%{y:,.2f}<extra></extra>" % donchian_window,
    ))
    fig.add_trace(go.Scatter(
        x=hist_opt.index, y=donchian_low, name=f"Mín. {donchian_window}r",
        line=dict(color="rgba(139,149,165,0.55)", width=1, dash="dot"),
        hovertemplate="Mín %d ruedas: %%{y:,.2f}<extra></extra>" % donchian_window,
    ))
    fig.add_trace(go.Candlestick(
        x=hist_opt.index, open=hist_opt["Open"], high=hist_opt["High"],
        low=hist_opt["Low"], close=hist_opt["Close"], name="Prima",
        increasing=dict(line=dict(color=TA_THEME["up"]), fillcolor=TA_THEME["up"]),
        decreasing=dict(line=dict(color=TA_THEME["down"]), fillcolor=TA_THEME["down"]),
    ))
    fig.add_trace(go.Scatter(
        x=hist_opt.index, y=sma_close, name=f"SMA {donchian_window}r", line=dict(color=TA_THEME["ema20"], width=1.4),
        hovertemplate="SMA: %{y:,.2f}<extra></extra>",
    ))
    if ppc is not None and not np.isnan(ppc):
        fig.add_hline(
            y=ppc, line_dash="dot", line_width=1.4, line_color="#f0b90b",
            annotation_text=f"PPC {ppc:,.2f}", annotation_position="right",
            annotation_font=dict(size=11, color="#f0b90b"), annotation_bgcolor="rgba(15,23,42,0.75)",
        )
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(tickformat=",.0f")
    return _finalize_layout(fig, 380, f"Histórico de la prima — {ticker} (Donchian {donchian_window}r + SMA)", "Prima (ARS)")


def option_history_stats(hist_opt, window=None):
    """
    Estadísticas del período: extremos tocados y dónde está el precio actual
    respecto de ellos. Con 'window' se limita a las últimas N ruedas (la
    misma ventana del Canal de Donchian), para una lectura de corto plazo de
    cuánto margen le queda a la prima para subir o bajar más -- no todo el
    historial desde que se emitió la serie.
    """
    if hist_opt is None or hist_opt.empty:
        return {}
    d = hist_opt.tail(int(window)) if window else hist_opt
    maximo = float(d["High"].max())
    minimo = float(d["Low"].min())
    ultimo = float(hist_opt["Close"].iloc[-1])
    rango = maximo - minimo
    pos_en_rango = (ultimo - minimo) / rango if rango > 0 else np.nan
    return {
        "maximo": maximo,
        "minimo": minimo,
        "ultimo": ultimo,
        "rango_pct": (rango / minimo * 100) if minimo > 0 else np.nan,
        "desde_maximo_pct": (ultimo / maximo - 1) * 100 if maximo > 0 else np.nan,
        "desde_minimo_pct": (ultimo / minimo - 1) * 100 if minimo > 0 else np.nan,
        "pos_en_rango_pct": pos_en_rango * 100 if not np.isnan(pos_en_rango) else np.nan,
        "n_ruedas": len(d),
    }


def normalize_portfolio(raw):
    """
    Parsea la respuesta de /api/v2/portafolio/{pais}. La estructura exacta de
    IOL no está 100% documentada acá, así que se prueban varias claves
    habituales (activos/titulo/tipo) de forma defensiva -- igual que se hace
    con normalize_options. Si algo no matchea, se puede revisar la respuesta
    cruda en el expander "Debug IOL" y ajustar las claves.
    """
    if isinstance(raw, dict):
        for key in ["activos", "data", "result", "items", "portafolio"]:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return pd.DataFrame()

    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        titulo = item.get("titulo") if isinstance(item.get("titulo"), dict) else {}
        simbolo = titulo.get("simbolo") or item.get("simbolo") or item.get("ticker") or ""
        tipo_activo = str(titulo.get("tipo") or item.get("tipo") or "").lower()

        rows.append({
            "Ticker": str(simbolo).upper(),
            "TipoActivo": tipo_activo,
            "Cantidad": clean_num(item.get("cantidad")),
            "PPC": clean_num(item.get("ppc") or item.get("precioPromedioCompra") or item.get("precioCompra")),
            "UltimoPrecio": clean_num(item.get("ultimoPrecio") or item.get("precioActual") or titulo.get("ultimoPrecio")),
            "Valorizado": clean_num(item.get("valorizado") or item.get("montoValorizado")),
            "GananciaPorcentaje": clean_num(item.get("gananciaPorcentaje") or item.get("variacionPorcentual")),
            "GananciaDinero": clean_num(item.get("gananciaDinero")),
        })
    return pd.DataFrame(rows)


def is_option_position(row):
    """Identifica si una fila de cartera es una opción de GGAL o YPF (las únicas que este panel sabe interpretar)."""
    ticker = str(row.get("Ticker", "")).upper()
    tipo_activo = str(row.get("TipoActivo", "")).lower()
    es_serie_conocida = ticker.startswith(("GFGC", "GFGV", "YPFC", "YPFV"))
    parece_opcion = "opc" in tipo_activo or es_serie_conocida
    return parece_opcion and es_serie_conocida


def underlying_for_option(ticker):
    ticker = str(ticker).upper()
    if ticker.startswith(("GFGC", "GFGV")):
        return "GGAL"
    if ticker.startswith(("YPFC", "YPFV")):
        return "YPF"
    return None


def portfolio_expiry_risk(resultados, detalles_extra):
    """
    Vista de riesgo de cartera por vencimiento (no por posición individual):
    cuánto valor nocional y cuánto valor extrínseco está concentrado en qué
    horizonte de tiempo, y cuánto se pierde/gana por día en toda la cartera
    sólo por el paso del tiempo (theta agregado). Sirve para ver si el
    riesgo de vencimiento está muy concentrado en pocos días, algo que mirar
    posición por posición no siempre deja ver.
    """
    if not resultados:
        return None

    filas = []
    theta_total = 0.0
    extrinsico_total = 0.0
    nocional_total = 0.0
    dias_lista = []

    for r in resultados:
        ticker = r["Ticker"]
        extra = detalles_extra.get(ticker, {})
        cantidad = clean_num(r.get("Cantidad"))
        ultimo = clean_num(r.get("Último"))
        dias = clean_num(extra.get("dias_venc_pos"))
        extrinsico = clean_num(extra.get("extrinsic_pos"))
        theta = clean_num(extra.get("theta_pos"))

        nocional = ultimo * cantidad if not np.isnan(ultimo) and not np.isnan(cantidad) else np.nan
        if not np.isnan(nocional):
            nocional_total += nocional
        if not np.isnan(extrinsico) and not np.isnan(cantidad):
            extrinsico_total += extrinsico * cantidad
        if not np.isnan(theta) and not np.isnan(cantidad):
            theta_total += theta * cantidad
        if not np.isnan(dias):
            dias_lista.append(dias)

        filas.append({"dias": dias, "nocional": nocional})

    dias_min = min(dias_lista) if dias_lista else np.nan

    buckets = {"≤5 días": 0.0, "6-15 días": 0.0, "16-30 días": 0.0, ">30 días": 0.0}
    for f in filas:
        d, n = f["dias"], f["nocional"]
        if np.isnan(d) or np.isnan(n):
            continue
        clave = "≤5 días" if d <= 5 else "6-15 días" if d <= 15 else "16-30 días" if d <= 30 else ">30 días"
        buckets[clave] += abs(n)

    return {
        "nocional_total": nocional_total,
        "extrinsico_total": extrinsico_total,
        "theta_total": theta_total,
        "dias_min": dias_min,
        "buckets": buckets,
    }


def position_expected_value(typ, K, cantidad, prima_actual, S, T, mu, hv):
    """
    Compara, para una posición ya abierta:
    - Valor de cerrarla ahora al precio de mercado actual (prima_actual * cantidad).
    - Valor esperado de mantenerla hasta el vencimiento, integrando el payoff
      bruto (sin descontar lo ya pagado, que es costo hundido) contra la
      misma distribución lognormal que usa el resto de la app (Advisor,
      Probabilidades). No es una garantía, es una comparación bajo el
      pronóstico vigente.
    """
    if K is None or np.isnan(K) or cantidad is None or np.isnan(cantidad) or T is None or T <= 0 or hv is None or np.isnan(hv) or hv <= 0:
        return np.nan, np.nan
    prices = np.linspace(max(S * 0.4, 1), S * 2.0, 400)
    payoff_bruto = np.maximum(prices - K, 0) if typ == "call" else np.maximum(K - prices, 0)
    weights = lognormal_weights(prices, S, T, mu, hv)
    area = _trapz(weights, prices)
    if area <= 0:
        return np.nan, np.nan
    weights_norm = weights / area
    valor_unitario = _trapz(weights_norm * payoff_bruto, prices)
    valor_esperado_mantener = float(valor_unitario * cantidad)
    valor_cierre_ahora = float(prima_actual * cantidad) if prima_actual is not None and not np.isnan(prima_actual) else np.nan
    return valor_esperado_mantener, valor_cierre_ahora


def recommend_option_action(typ, cantidad, dias_venc, extrinsic, prima_actual, ppc, prob_dict, veredicto_tecnico,
                             valor_esperado_mantener=None, valor_cierre_ahora=None, liquidez=None):
    """
    Recomendación simple y transparente (MANTENER / VIGILAR / VENDER) para una
    posición de opciones en cartera. Combina:
    - Hacia dónde favorece el pronóstico Busa AI y el veredicto técnico del
      subyacente, según si la posición es un call o un put (y si está
      comprada o vendida/lanzada).
    - Cuánto tiempo queda al vencimiento.
    - Cuánto valor extrínseco le queda (si ya no tiene casi nada, poco
      sentido tiene esperar más).
    - Valor esperado de mantener hasta el vencimiento vs. cerrar ahora
      (integrando contra el mismo modelo lognormal que usa el Advisor).
    - Liquidez de esta opción puntual (no del papel): si es muy baja, cuesta
      más salir a buen precio.

    No es asesoramiento financiero personalizado: es una lectura basada en
    reglas explícitas, pensada para acompañar la decisión, no reemplazarla.
    """
    razones = []
    score = 0.0

    vendida = cantidad is not None and not np.isnan(cantidad) and cantidad < 0
    favorable = "Sube" if typ == "call" else "Baja"
    contraria = "Baja" if typ == "call" else "Sube"
    if vendida:
        favorable, contraria = contraria, favorable
        razones.append("Posición vendida/lanzada: la lectura de favorable/desfavorable está invertida respecto de una posición comprada.")

    p_fav = float(prob_dict.get(favorable, 0))
    p_con = float(prob_dict.get(contraria, 0))
    if p_fav > p_con + 0.10:
        score += 1
        razones.append(f"Busa AI favorece {favorable.lower()} ({p_fav:.0%} vs {p_con:.0%}), a favor de la posición.")
    elif p_con > p_fav + 0.10:
        score -= 1
        razones.append(f"Busa AI favorece {contraria.lower()} ({p_con:.0%} vs {p_fav:.0%}), en contra de la posición.")
    else:
        razones.append("Busa AI está parejo entre ambos escenarios, sin sesgo claro.")

    favorable_verdict = "SUBE" if favorable == "Sube" else "BAJA"
    contraria_verdict = "BAJA" if favorable == "Sube" else "SUBE"
    if veredicto_tecnico == favorable_verdict:
        score += 1
        razones.append(f"El veredicto técnico también es {veredicto_tecnico} (a favor).")
    elif veredicto_tecnico == contraria_verdict:
        score -= 1
        razones.append(f"El veredicto técnico es {veredicto_tecnico} (en contra de la posición).")
    else:
        razones.append("El veredicto técnico está LATERAL, sin señal direccional clara.")

    if dias_venc is not None and not np.isnan(dias_venc):
        if dias_venc <= 5:
            score -= 0.75
            razones.append(f"Quedan {int(dias_venc)} días para el vencimiento: poco margen para que la tesis se termine de confirmar.")
        elif dias_venc <= 15:
            score -= 0.25
            razones.append(f"Quedan {int(dias_venc)} días para el vencimiento: empieza a correr el tiempo.")

    if extrinsic is not None and not np.isnan(extrinsic) and not np.isnan(prima_actual) and prima_actual > 0:
        if extrinsic <= 0.05 * prima_actual:
            score -= 0.5
            razones.append("Casi no le queda valor extrínseco: ya se movió casi todo lo que tenía para moverse por prima de tiempo.")

    if valor_esperado_mantener is not None and valor_cierre_ahora is not None and not np.isnan(valor_esperado_mantener) and not np.isnan(valor_cierre_ahora):
        base = abs(valor_cierre_ahora) if valor_cierre_ahora != 0 else abs(valor_esperado_mantener)
        if base > 0:
            diff_pct = (valor_esperado_mantener - valor_cierre_ahora) / base * 100
            if diff_pct > 8:
                score += 1
                razones.append(f"El valor esperado de mantener hasta el vencimiento ({valor_esperado_mantener:,.0f}) supera al de cerrar ahora ({valor_cierre_ahora:,.0f}), según el pronóstico vigente.")
            elif diff_pct < -8:
                score -= 1
                razones.append(f"Cerrar ahora ({valor_cierre_ahora:,.0f}) rinde más que el valor esperado de mantener ({valor_esperado_mantener:,.0f}), según el pronóstico vigente.")
            else:
                razones.append("El valor esperado de mantener y el de cerrar ahora son similares.")

    if liquidez is not None and not np.isnan(liquidez) and liquidez < 35:
        score -= 0.25
        razones.append(f"Liquidez baja en esta opción puntual ({liquidez:.0f}/100): podría costar salir a buen precio.")

    pnl_pct = np.nan
    if ppc is not None and not np.isnan(ppc) and ppc > 0 and prima_actual is not None and not np.isnan(prima_actual):
        pnl_pct = (prima_actual - ppc) / ppc * 100
        if vendida:
            pnl_pct = -pnl_pct

    if score >= 1.25:
        accion = "MANTENER"
    elif score <= -1.25:
        accion = "VENDER"
    else:
        accion = "VIGILAR"

    return accion, score, razones, pnl_pct


def analyze(df, S, T, r, hv, p_up, p_down, mode):
    rows = []
    favorites = set(load_favorites())
    hoy = datetime.now().date()
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

        # Vencimiento automático por ticker (tercer viernes del mes que indican
        # las últimas 2 letras). Si no se puede inferir, cae al valor manual
        # de 'Días vencimiento' de la barra lateral (T ya viene calculado con eso).
        fecha_venc, dias_venc, fuente_venc = parse_option_expiry(row.get("Ticker"), hoy)
        if fecha_venc is not None:
            T_row = max(dias_venc, 0) / 365
        else:
            T_row = T
            dias_venc = int(round(T * 365))

        theo = bs_price(S, K, T_row, r, hv, typ)
        iv = implied_vol(prima, S, K, T_row, r, typ) if not np.isnan(prima) else np.nan
        sig = iv if not np.isnan(iv) else hv
        delta, gamma, vega, theta, rho, prob_itm = greeks(S, K, T_row, r, sig, typ)
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
            "Vol. Compra": clean_num(row.get("Vol. Compra")),
            "Vol. Venta": clean_num(row.get("Vol. Venta")),
            "Último": ultimo,
            "Vencimiento": fecha_venc.strftime("%d/%m/%Y") if fecha_venc is not None else "",
            "Días venc.": dias_venc,
            "Fuente venc.": "Ticker (3er viernes)" if fuente_venc == "ticker" else "Manual (barra lateral)",
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
    for c in ["Strike", "Score Busa", "Volumen", "Vol. Compra", "Vol. Venta", "Días venc."]:
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
# Indicadores técnicos clásicos (RSI, MACD, Bollinger)
# =========================
def compute_rsi(close, period=14):
    """RSI de Wilder (suavizado exponencial), el estándar de la mayoría de las plataformas."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_bollinger(close, period=20, num_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma, sma + num_std * std, sma - num_std * std


def compute_adx(high, low, close, period=14):
    """
    ADX(14) de Wilder + DI/-DI. A diferencia de RSI/MACD (que miden dirección
    y momentum), el ADX mide la FUERZA de la tendencia -- clave para saber si
    conviene confiar en las señales direccionales o si el papel está en un
    rango sin tendencia real.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def rsi_expert_reading(rsi_series):
    s = rsi_series.dropna()
    if s.empty:
        return "Sin datos suficientes para calcular RSI."
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else last
    if last >= 70:
        zona = "sobrecompra"
    elif last <= 30:
        zona = "sobreventa"
    elif last >= 50:
        zona = "neutral-alcista"
    else:
        zona = "neutral-bajista"
    direccion = "subiendo" if last > prev + 0.05 else "bajando" if last < prev - 0.05 else "estable"
    return f"RSI(14) en {last:.1f} — zona {zona}, {direccion} respecto de la rueda anterior ({prev:.1f})."

def macd_expert_reading(macd_series, signal_series, hist_series):
    h = hist_series.dropna()
    if h.empty:
        return "Sin datos suficientes para calcular MACD."
    m, s_val, hist = float(macd_series.iloc[-1]), float(signal_series.iloc[-1]), float(h.iloc[-1])
    hist_prev = float(h.iloc[-2]) if len(h) > 1 else hist
    estado = "por encima de la señal (sesgo comprador)" if m > s_val else "por debajo de la señal (sesgo vendedor)"
    if abs(hist) > abs(hist_prev) + 0.01:
        momentum = "el histograma se está expandiendo: el momentum actual se refuerza"
    elif abs(hist) < abs(hist_prev) - 0.01:
        momentum = "el histograma se está achicando: el momentum actual pierde fuerza, posible cruce cercano"
    else:
        momentum = "el histograma está estable"
    return f"MACD {estado}; {momentum}."

def bollinger_expert_reading(close_series, sma, upper, lower):
    u = upper.dropna()
    if u.empty:
        return "Sin datos suficientes para calcular Bandas de Bollinger."
    price = float(close_series.iloc[-1])
    u_val, l_val, mid = float(upper.iloc[-1]), float(lower.iloc[-1]), float(sma.iloc[-1])
    width_pct = (u_val - l_val) / mid * 100 if mid else np.nan
    pos = (price - l_val) / (u_val - l_val) if (u_val - l_val) else np.nan
    if pos >= 0.95:
        zona = "tocando la banda superior (posible sobre-extensión de corto plazo)"
    elif pos <= 0.05:
        zona = "tocando la banda inferior (posible sobre-extensión bajista)"
    else:
        zona = f"al {pos*100:.0f}% del ancho de la banda (0=piso, 100=techo)"
    if width_pct < 8:
        compresion = "banda comprimida: la volatilidad está baja, atenti a una ruptura próxima"
    elif width_pct > 20:
        compresion = "banda expandida: volatilidad alta"
    else:
        compresion = "ancho de banda moderado"
    return f"Precio {zona}. {compresion} ({width_pct:.1f}% del precio medio)."

def trend_expert_reading(close_series, ema50):
    e = ema50.dropna()
    if e.empty:
        return "Sin datos suficientes para evaluar la tendencia de fondo."
    price = float(close_series.iloc[-1])
    ema50_val = float(ema50.iloc[-1])
    if price > ema50_val:
        return f"Precio por encima de la EMA50 ({ema50_val:,.2f}) — estructura de tendencia alcista de mediano plazo."
    return f"Precio por debajo de la EMA50 ({ema50_val:,.2f}) — estructura de tendencia bajista de mediano plazo."

def volume_expert_reading(volume_series, close_series):
    if volume_series is None or volume_series.dropna().empty:
        return "Sin datos de volumen disponibles."
    vol = volume_series.dropna()
    if len(vol) < 21:
        return "Historial de volumen insuficiente para comparar contra el promedio."
    vol_avg20 = vol.rolling(20).mean()
    vol_last = float(vol.iloc[-1])
    vol_avg_last = float(vol_avg20.iloc[-1])
    if not vol_avg_last:
        return "No pude comparar el volumen contra su promedio."
    ratio = vol_last / vol_avg_last
    price_change = float(close_series.iloc[-1]) - float(close_series.iloc[-2])
    nivel = "muy por encima" if ratio >= 1.5 else "por encima" if ratio >= 1.2 else "por debajo" if ratio <= 0.8 else "en línea con"
    direccion = "suba" if price_change > 0 else "baja" if price_change < 0 else "sin cambio"
    confirma = ""
    if ratio >= 1.2 and price_change != 0:
        confirma = " El volumen elevado le da más peso a este movimiento: hay convicción real detrás del precio."
    elif ratio <= 0.8:
        confirma = " Volumen flojo: el movimiento reciente podría no tener demasiada convicción detrás todavía."
    return f"Volumen {nivel} su promedio de 20 ruedas (ratio {ratio:.2f}x), en una rueda de {direccion}.{confirma}"

def adx_expert_reading(adx, plus_di, minus_di):
    a = adx.dropna()
    if a.empty:
        return "Sin datos suficientes para calcular ADX."
    adx_val = float(a.iloc[-1])
    pdi, mdi = float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
    if adx_val >= 40:
        fuerza = "tendencia muy fuerte"
    elif adx_val >= 25:
        fuerza = "tendencia con fuerza real"
    elif adx_val >= 20:
        fuerza = "tendencia incipiente, todavía débil"
    else:
        fuerza = "sin tendencia definida: mercado lateral/en rango"
    direccion = "alcista (+DI por encima de -DI)" if pdi > mdi else "bajista (-DI por encima de +DI)"
    return f"ADX(14) en {adx_val:.1f}: {fuerza}. Dirección dominante {direccion} (+DI {pdi:.1f} / -DI {mdi:.1f})."


def technical_verdict(rsi, macd, macd_signal, close_series, sma20, ema50, volume_series, adx, plus_di, minus_di):
    """
    Veredicto técnico consolidado (Sube/Baja/Lateral): una votación simple y
    transparente entre RSI, MACD, posición vs. el centro de Bollinger y
    tendencia (EMA50), con el volumen como confirmación (medio voto, no voto
    completo, porque el volumen no tiene dirección propia). El ADX no vota:
    modula qué tanta confianza darle al veredicto (si no hay tendencia real,
    el veredicto direccional vale menos).

    Es independiente del pronóstico estadístico de Busa AI (que usa un modelo
    lognormal con aprendizaje bayesiano) -- pueden coincidir o no.
    """
    score = 0.0
    detalle = []

    rsi_s = rsi.dropna()
    if not rsi_s.empty:
        rsi_last = float(rsi_s.iloc[-1])
        if rsi_last > 55:
            score += 1; detalle.append(("RSI", "alcista"))
        elif rsi_last < 45:
            score -= 1; detalle.append(("RSI", "bajista"))
        else:
            detalle.append(("RSI", "neutral"))

    macd_s = macd.dropna()
    if not macd_s.empty:
        if float(macd.iloc[-1]) > float(macd_signal.iloc[-1]):
            score += 1; detalle.append(("MACD", "alcista"))
        else:
            score -= 1; detalle.append(("MACD", "bajista"))

    sma_s = sma20.dropna()
    if not sma_s.empty:
        price = float(close_series.iloc[-1])
        if price > float(sma20.iloc[-1]):
            score += 1; detalle.append(("Bollinger (vs. centro)", "alcista"))
        else:
            score -= 1; detalle.append(("Bollinger (vs. centro)", "bajista"))

    ema_s = ema50.dropna()
    if not ema_s.empty:
        price = float(close_series.iloc[-1])
        if price > float(ema50.iloc[-1]):
            score += 1; detalle.append(("Tendencia (EMA50)", "alcista"))
        else:
            score -= 1; detalle.append(("Tendencia (EMA50)", "bajista"))

    if volume_series is not None and len(volume_series.dropna()) > 20:
        vol_avg20 = volume_series.rolling(20).mean()
        vol_last = float(volume_series.iloc[-1])
        vol_avg_last = float(vol_avg20.iloc[-1])
        price_change = float(close_series.iloc[-1]) - float(close_series.iloc[-2])
        if vol_avg_last and vol_last > vol_avg_last * 1.2:
            if price_change > 0:
                score += 0.5; detalle.append(("Volumen", "confirma alcista"))
            elif price_change < 0:
                score -= 0.5; detalle.append(("Volumen", "confirma bajista"))

    adx_s = adx.dropna()
    adx_last = float(adx_s.iloc[-1]) if not adx_s.empty else None
    tendencia_fuerte = adx_last is not None and adx_last >= 25

    if score >= 1.5:
        veredicto = "SUBE"
    elif score <= -1.5:
        veredicto = "BAJA"
    else:
        veredicto = "LATERAL"

    return veredicto, score, detalle, adx_last, tendencia_fuerte


# =========================
# Gráficos individuales (uno por indicador, cada uno con su propia lectura)
# =========================
TA_THEME = {
    "up": "#26a69a",
    "down": "#ef5350",
    "price": "#e5e7eb",
    "ema20": "#f0b90b",
    "ema50": "#3b82f6",
    "band": "#8b95a5",
    "band_fill": "rgba(139,149,165,0.08)",
    "rsi": "#c084fc",
    "macd": "#3b82f6",
    "signal": "#f0b90b",
    "adx": "#f0b90b",
    "plus_di": "#26a69a",
    "minus_di": "#ef5350",
    "vol_avg": "#f0b90b",
    "grid": "rgba(255,255,255,0.055)",
    "zeroline": "rgba(255,255,255,0.15)",
    "text": "#cbd5e1",
    "title": "#f1f5f9",
}

def _style_axes(fig, yaxis_title=None):
    fig.update_xaxes(showgrid=True, gridcolor=TA_THEME["grid"], zeroline=False, showline=True, linecolor=TA_THEME["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=TA_THEME["grid"], zeroline=False, showline=True, linecolor=TA_THEME["grid"], title_text=yaxis_title, title_font=dict(size=11, color=TA_THEME["text"]))
    return fig

def _finalize_layout(fig, height, title, yaxis_title=None, show_legend=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Roboto, Arial, sans-serif", size=12, color=TA_THEME["text"]),
        title=dict(text=title, font=dict(size=14, color=TA_THEME["title"]), x=0.01, xanchor="left", y=0.99, yanchor="top"),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1f2937", font_size=12, font_family="Segoe UI, Arial, sans-serif", bordercolor="rgba(255,255,255,0.1)"),
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="top", y=0.88, x=0,
            bgcolor="rgba(15,23,42,0.55)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=55, t=80, b=25),
    )
    return _style_axes(fig, yaxis_title)

def _last_value_line(fig, x_last, y_last, color, fmt="{:,.2f}"):
    fig.add_hline(
        y=y_last, line_dash="dot", line_width=1, line_color="rgba(255,255,255,0.35)",
        annotation_text=fmt.format(y_last), annotation_position="right",
        annotation_font=dict(size=11, color=color), annotation_bgcolor="rgba(15,23,42,0.75)",
    )
    return fig


def build_price_figure(hist, sma20, bb_up, bb_dn, ema20, ema50, ticker_key="", lookback_days=180):
    idx = hist.index[-lookback_days:] if len(hist) > lookback_days else hist.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=bb_up.loc[idx], name="Banda sup. (20,2)",
        line=dict(color=TA_THEME["band"], width=1, dash="dot"),
        hovertemplate="Banda sup.: %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=idx, y=bb_dn.loc[idx], name="Banda inf. (20,2)",
        line=dict(color=TA_THEME["band"], width=1, dash="dot"),
        fill="tonexty", fillcolor=TA_THEME["band_fill"],
        hovertemplate="Banda inf.: %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Candlestick(
        x=idx, open=hist.loc[idx, "Open"], high=hist.loc[idx, "High"],
        low=hist.loc[idx, "Low"], close=hist.loc[idx, "Close"],
        name="Precio",
        increasing=dict(line=dict(color=TA_THEME["up"]), fillcolor=TA_THEME["up"]),
        decreasing=dict(line=dict(color=TA_THEME["down"]), fillcolor=TA_THEME["down"]),
    ))
    fig.add_trace(go.Scatter(x=idx, y=ema20.loc[idx], name="EMA20", line=dict(color=TA_THEME["ema20"], width=1.5),
                              hovertemplate="EMA20: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=ema50.loc[idx], name="EMA50", line=dict(color=TA_THEME["ema50"], width=1.5),
                              hovertemplate="EMA50: %{y:,.2f}<extra></extra>"))

    last_price = float(hist.loc[idx, "Close"].iloc[-1])
    _last_value_line(fig, idx[-1], last_price, TA_THEME["price"])

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(tickformat=",.0f")
    return _finalize_layout(fig, 460, f"Precio, Bollinger(20,2) y EMAs — {ticker_key}", "Precio (ARS)")

def build_volume_figure(hist, ticker_key="", lookback_days=180):
    idx = hist.index[-lookback_days:] if len(hist) > lookback_days else hist.index
    vol = hist.loc[idx, "Volume"]
    colors = [TA_THEME["up"] if c >= o else TA_THEME["down"] for c, o in zip(hist.loc[idx, "Close"], hist.loc[idx, "Open"])]
    vol_avg = hist["Volume"].rolling(20).mean().loc[idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=vol, marker_color=colors, opacity=0.75, name="Volumen",
                          hovertemplate="Volumen: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=vol_avg, name="Promedio 20 ruedas", line=dict(color=TA_THEME["vol_avg"], width=1.6),
                              hovertemplate="Promedio 20r: %{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickformat=",.0f")
    return _finalize_layout(fig, 260, f"Volumen — {ticker_key}", "Volumen (nominales)")

def build_rsi_figure(rsi, ticker_key="", lookback_days=180):
    idx = rsi.index[-lookback_days:] if len(rsi) > lookback_days else rsi.index
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor=TA_THEME["down"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=TA_THEME["up"], opacity=0.06, line_width=0)
    fig.add_trace(go.Scatter(x=idx, y=rsi.loc[idx], name="RSI(14)", line=dict(color=TA_THEME["rsi"], width=2),
                              hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=70, line_dash="dash", line_width=1, line_color=TA_THEME["down"], annotation_text="Sobrecompra 70", annotation_font=dict(size=10, color=TA_THEME["down"]))
    fig.add_hline(y=30, line_dash="dash", line_width=1, line_color=TA_THEME["up"], annotation_text="Sobreventa 30", annotation_font=dict(size=10, color=TA_THEME["up"]))
    fig.add_hline(y=50, line_dash="dot", line_width=1, line_color=TA_THEME["zeroline"])
    last_rsi = float(rsi.loc[idx].dropna().iloc[-1]) if not rsi.loc[idx].dropna().empty else None
    if last_rsi is not None:
        _last_value_line(fig, idx[-1], last_rsi, TA_THEME["rsi"], fmt="{:.1f}")
    fig.update_yaxes(range=[0, 100])
    return _finalize_layout(fig, 260, f"RSI(14) — {ticker_key}", "RSI", show_legend=False)

def build_macd_figure(macd, macd_signal, macd_hist, ticker_key="", lookback_days=180):
    idx = macd.index[-lookback_days:] if len(macd) > lookback_days else macd.index
    hist_vals = macd_hist.loc[idx]
    colors = [TA_THEME["up"] if v >= 0 else TA_THEME["down"] for v in hist_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=hist_vals, name="Histograma", marker_color=colors, opacity=0.6,
                          hovertemplate="Histograma: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=macd.loc[idx], name="MACD", line=dict(color=TA_THEME["macd"], width=2),
                              hovertemplate="MACD: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=macd_signal.loc[idx], name="Señal", line=dict(color=TA_THEME["signal"], width=1.5),
                              hovertemplate="Señal: %{y:,.2f}<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color=TA_THEME["zeroline"])
    return _finalize_layout(fig, 300, f"MACD(12,26,9) — {ticker_key}", "MACD")

def build_adx_figure(adx, plus_di, minus_di, ticker_key="", lookback_days=180):
    idx = adx.index[-lookback_days:] if len(adx) > lookback_days else adx.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=adx.loc[idx], name="ADX(14)", line=dict(color=TA_THEME["adx"], width=2.2),
                              hovertemplate="ADX: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=plus_di.loc[idx], name="+DI", line=dict(color=TA_THEME["plus_di"], width=1.3),
                              hovertemplate="+DI: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=minus_di.loc[idx], name="-DI", line=dict(color=TA_THEME["minus_di"], width=1.3),
                              hovertemplate="-DI: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=25, line_dash="dash", line_width=1, line_color=TA_THEME["band"], annotation_text="Umbral de tendencia (25)", annotation_font=dict(size=10, color=TA_THEME["band"]))
    return _finalize_layout(fig, 280, f"ADX(14) — Fuerza de tendencia — {ticker_key}", "ADX / DI")


def render_technical_panel(ticker_key, activo_seleccionado, period, horizon, lateral, lookback,
                            drift_shrink, use_tilt, tilt_strength, learning_window, learning_prior_strength):
    """
    Panel completo estilo 'experto en análisis técnico' para un ticker:
    pronóstico Busa AI (Sube/Baja/Lateral) + veredicto técnico consolidado +
    un gráfico independiente por indicador (precio/Bollinger/EMAs, volumen,
    RSI, MACD, ADX), cada uno con su propia lectura. Se usa una vez por cada
    activo (GGAL e YPF), independiente de cuál esté elegido en la barra lateral.
    """
    hist = get_hist(TICKERS[ticker_key]["local"], period)
    if hist.empty:
        st.warning(f"No pude descargar histórico de {ticker_key}.")
        return

    close_series = hist["Close"].dropna()
    volume_series = hist["Volume"] if "Volume" in hist.columns else None
    S_local = float(close_series.iloc[-1])
    fuente_precio = "yfinance"
    if ticker_key == activo_seleccionado and "spot_iol" in st.session_state:
        S_local = float(st.session_state["spot_iol"])
        fuente_precio = "IOL"

    prob_local = prob_data(hist, int(horizon), lateral, int(lookback), drift_shrink, use_tilt, tilt_strength)
    prob_local = apply_learning_to_probabilities(prob_local, ticker_key, int(learning_window), int(learning_prior_strength))

    rsi = compute_rsi(close_series)
    macd, macd_signal, macd_hist = compute_macd(close_series)
    sma20, bb_up, bb_dn = compute_bollinger(close_series)
    ema20 = close_series.ewm(span=20, adjust=False).mean()
    ema50 = close_series.ewm(span=50, adjust=False).mean()
    adx, plus_di, minus_di = compute_adx(hist["High"], hist["Low"], close_series)

    # --- Pronóstico Busa AI (modelo estadístico) ---
    with st.container(border=True):
        st.markdown(f"##### Pronóstico Busa AI — {ticker_key}")
        prob_bar("Sube", prob_local["Sube"], "#15803d")
        prob_bar("Baja", prob_local["Baja"], "#b91c1c")
        prob_bar("Lateral", prob_local["Lateral"], "#ca8a04")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"{S_local:,.2f}")
        c2.metric("Nivel suba", f"{prob_local['Nivel suba']:,.2f}")
        c3.metric("Nivel baja", f"{prob_local['Nivel baja']:,.2f}")
        generado = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        ultimo_dato = pd.to_datetime(close_series.index[-1]).strftime("%d/%m/%Y")
        st.caption(f"🕒 Predicción generada: {generado} hs. | Datos de precio hasta: {ultimo_dato} | Fuente del precio: {fuente_precio} | Horizonte: {int(horizon)} ruedas.")

    # --- Hoy en vivo (IOL): el gráfico de abajo es de cierres diarios (yfinance)
    # y por diseño no incluye la rueda de hoy hasta que cierra. Esto sí trae
    # el dato de la rueda de hoy en curso, directo de IOL. ---
    with st.container(border=True):
        st.markdown(f"##### Hoy en vivo (IOL) — {ticker_key}")
        quote_hoy = None
        if ticker_key == activo_seleccionado and "quote_iol" in st.session_state:
            quote_hoy = st.session_state["quote_iol"]
        else:
            try:
                client_hoy = IOLClient.from_config()
                quote_hoy = client_hoy.get_quote(TICKERS[ticker_key]["iol"])
            except Exception:
                quote_hoy = None

        campos_hoy = extract_option_quote_fields(quote_hoy) if quote_hoy else {}
        if not campos_hoy or (np.isnan(clean_num(campos_hoy.get("Último"))) and np.isnan(clean_num(campos_hoy.get("Volumen")))):
            st.info("No pude traer la cotización en vivo de hoy (puede ser que el mercado esté cerrado o falte autenticación IOL).")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Último", "" if pd.isna(clean_num(campos_hoy.get("Último"))) else f"{campos_hoy['Último']:,.2f}")
            h2.metric("Variación", "" if pd.isna(clean_num(campos_hoy.get("Variación %"))) else f"{campos_hoy['Variación %']:.2f}%")
            h3.metric("Apertura", "" if pd.isna(clean_num(campos_hoy.get("Apertura"))) else f"{campos_hoy['Apertura']:,.2f}")
            h4.metric("Máx. / Mín. día", "" if pd.isna(clean_num(campos_hoy.get("Máximo"))) else f"{campos_hoy['Máximo']:,.2f} / {campos_hoy.get('Mínimo', float('nan')):,.2f}")
            h5, h6, h7 = st.columns(3)
            h5.metric("Volumen operado hoy", "" if pd.isna(clean_num(campos_hoy.get("Volumen"))) else f"{campos_hoy['Volumen']:,.0f}")
            h6.metric("Vol. Compra (puntas)", "" if pd.isna(clean_num(campos_hoy.get("Vol. Compra"))) else f"{campos_hoy['Vol. Compra']:,.0f}")
            h7.metric("Vol. Venta (puntas)", "" if pd.isna(clean_num(campos_hoy.get("Vol. Venta"))) else f"{campos_hoy['Vol. Venta']:,.0f}")
            st.caption(f"🕒 Consultado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} hs. 'Vol. Compra/Venta' es la cantidad ofrecida en las puntas ahora mismo, no un desglose de operaciones ejecutadas por comprador/vendedor (IOL no expone esa apertura).")

        # Opciones de este subyacente que estén en la cartera ya cargada
        # (Favoritos → Mi cartera IOL), con su volumen en vivo.
        port_df_tech = st.session_state.get("portfolio_df", pd.DataFrame())
        if not port_df_tech.empty:
            opciones_cartera_tech = port_df_tech[port_df_tech.apply(is_option_position, axis=1)].copy()
            opciones_cartera_tech = opciones_cartera_tech[opciones_cartera_tech["Ticker"].apply(underlying_for_option) == ticker_key]
            if not opciones_cartera_tech.empty:
                st.markdown(f"**Tus opciones de {ticker_key} en cartera — volumen de hoy**")
                try:
                    client_opts_tech = IOLClient.from_config()
                except Exception:
                    client_opts_tech = None
                filas_opt_vol = []
                for _, pos_tech in opciones_cartera_tech.iterrows():
                    tk_tech = pos_tech["Ticker"]
                    campos_opt_tech = {}
                    if client_opts_tech is not None:
                        try:
                            campos_opt_tech = extract_option_quote_fields(client_opts_tech.get_quote(tk_tech))
                        except Exception:
                            campos_opt_tech = {}
                    filas_opt_vol.append({
                        "Ticker": tk_tech,
                        "Último": campos_opt_tech.get("Último", np.nan),
                        "Volumen operado hoy": campos_opt_tech.get("Volumen", np.nan),
                        "Vol. Compra": campos_opt_tech.get("Vol. Compra", np.nan),
                        "Vol. Venta": campos_opt_tech.get("Vol. Venta", np.nan),
                    })
                st.dataframe(
                    pd.DataFrame(filas_opt_vol).style.format(
                        {"Último": "{:.2f}", "Volumen operado hoy": "{:,.0f}", "Vol. Compra": "{:,.0f}", "Vol. Venta": "{:,.0f}"}, na_rep="",
                    ),
                    use_container_width=True,
                )
            else:
                st.caption(f"No tenés opciones de {ticker_key} cargadas en 'Mi cartera (IOL)' (pestaña Favoritos).")
        else:
            st.caption("Cargá tu cartera en Favoritos → 'Mi cartera (IOL)' para ver acá también el volumen de tus opciones puntuales.")

    # --- Veredicto técnico consolidado ---
    veredicto, score, detalle, adx_last, tendencia_fuerte = technical_verdict(
        rsi, macd, macd_signal, close_series, sma20, ema50, volume_series, adx, plus_di, minus_di
    )
    css_class = "score-good" if veredicto == "SUBE" else "score-bad" if veredicto == "BAJA" else "score-mid"
    icono = "▲" if veredicto == "SUBE" else "▼" if veredicto == "BAJA" else "➡"
    with st.container(border=True):
        st.markdown(f"##### Veredicto técnico — {ticker_key}")
        st.markdown(f'<span class="{css_class}" style="font-size:26px;">{icono} {veredicto}</span>', unsafe_allow_html=True)
        detalle_txt = " · ".join([f"{k}: {v}" for k, v in detalle])
        st.caption(f"Puntaje: {score:+.1f} (rango -4.5 a +4.5) — {detalle_txt}")
        if adx_last is not None:
            conf_txt = "tendencia confirmada (ADX ≥ 25): el veredicto tiene más respaldo" if tendencia_fuerte else "sin tendencia fuerte todavía (ADX < 25): tomar el veredicto con más cautela, el mercado puede estar en rango"
            st.caption(f"Fuerza de tendencia (ADX): {adx_last:.1f} — {conf_txt}.")
        st.caption("Veredicto técnico (RSI + MACD + Bollinger + tendencia + volumen). Es independiente del pronóstico estadístico Busa AI de arriba: pueden coincidir o no.")

    # --- Gráfico 1: Precio + Bollinger + EMAs ---
    with st.container(border=True):
        st.write(f"📉 {bollinger_expert_reading(close_series, sma20, bb_up, bb_dn)}")
        st.write(f"📐 {trend_expert_reading(close_series, ema50)}")
        st.plotly_chart(build_price_figure(hist, sma20, bb_up, bb_dn, ema20, ema50, ticker_key), use_container_width=True, config={"displaylogo": False})

    # --- Gráfico 2: Volumen ---
    with st.container(border=True):
        st.write(f"📦 {volume_expert_reading(volume_series, close_series)}")
        st.plotly_chart(build_volume_figure(hist, ticker_key), use_container_width=True, config={"displaylogo": False})

    # --- Gráfico 3: RSI ---
    with st.container(border=True):
        st.write(f"📈 {rsi_expert_reading(rsi)}")
        st.plotly_chart(build_rsi_figure(rsi, ticker_key), use_container_width=True, config={"displaylogo": False})

    # --- Gráfico 4: MACD ---
    with st.container(border=True):
        st.write(f"📊 {macd_expert_reading(macd, macd_signal, macd_hist)}")
        st.plotly_chart(build_macd_figure(macd, macd_signal, macd_hist, ticker_key), use_container_width=True, config={"displaylogo": False})

    # --- Gráfico 5: ADX (fuerza de tendencia) ---
    with st.container(border=True):
        st.write(f"🧭 {adx_expert_reading(adx, plus_di, minus_di)}")
        st.plotly_chart(build_adx_figure(adx, plus_di, minus_di, ticker_key), use_container_width=True, config={"displaylogo": False})

    st.caption("Análisis técnico educativo (RSI, MACD, Bollinger, EMAs, Volumen, ADX). No constituye recomendación financiera personalizada.")


# =========================
# Motor de payoff (usado por Advisor)
# =========================
def first_valid_price(row):
    for col in ["Prima usada", "Último", "Venta", "Compra"]:
        val = clean_num(row.get(col))
        if not np.isnan(val) and val > 0:
            return float(val)
    return np.nan


def _trapz(y, x):
    """
    Integración trapezoidal compatible con numpy nuevo y viejo:
    numpy >= 2.0 renombró np.trapz a np.trapezoid y eliminó el alias viejo.
    Como requirements.txt no fija versión de numpy, esto evita que la app
    se rompa según qué versión instale Streamlit Cloud.
    """
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return fn(y, x)


def lognormal_weights(prices, S, T, mu, hv):
    """
    Densidad de precio al vencimiento bajo el mismo modelo lognormal (GBM)
    usado para las probabilidades Sube/Baja/Lateral, evaluada en una grilla
    de precios. Se usa para puntuar estrategias con el mismo criterio
    estadístico que el pronóstico, en vez de una heurística aparte.
    """
    prices = np.asarray(prices, dtype=float)
    if T <= 0 or hv <= 0 or S <= 0:
        return np.zeros_like(prices)
    mean_log = np.log(S) + (mu - 0.5 * hv**2) * T
    sd = hv * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = norm.pdf((np.log(prices) - mean_log) / sd) / (prices * sd)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    return pdf


def strategy_probability_and_ev(payoff, prices, S, T, mu, hv):
    """
    Probabilidad de éxito (payoff > 0) y valor esperado de una estrategia,
    integrando el payoff contra la densidad lognormal del pronóstico vigente.
    Reemplaza la estimación anterior (heurística lineal por distancia al
    break-even) por un cálculo consistente con el modelo de probabilidades.
    """
    weights = lognormal_weights(prices, S, T, mu, hv)
    area = _trapz(weights, prices)
    if area <= 0:
        return np.nan, np.nan
    weights_norm = weights / area
    prob_profit = float(_trapz(weights_norm * (payoff > 0), prices))
    expected_value = float(_trapz(weights_norm * payoff, prices))
    return float(np.clip(prob_profit, 0.0, 1.0)), expected_value


def _liquidity_score_ticker(ticker, analyzed_df):
    """
    Puntaje 0-100 de "operabilidad" de una pata según volumen y spread
    compra/venta. 50 = neutral (sin datos o pata no usada).
    """
    if not ticker or analyzed_df is None or analyzed_df.empty:
        return 50.0
    row = analyzed_df[analyzed_df["Ticker"] == ticker]
    if row.empty:
        return 50.0
    row = row.iloc[0]
    vol = clean_num(row.get("Volumen"))
    compra = clean_num(row.get("Compra"))
    venta = clean_num(row.get("Venta"))
    score = 50.0
    if not np.isnan(vol):
        if vol >= 500:
            score += 25
        elif vol >= 100:
            score += 10
        elif vol <= 5:
            score -= 20
    if not np.isnan(compra) and not np.isnan(venta) and venta > 0:
        spread_pct = (venta - compra) / venta * 100
        if spread_pct <= 3:
            score += 25
        elif spread_pct <= 8:
            score += 10
        elif spread_pct >= 20:
            score -= 25
    return float(np.clip(score, 0, 100))


def liquidity_score_for_legs(tickers, analyzed_df):
    ts = [t for t in tickers if t]
    if not ts:
        return 50.0
    return float(np.mean([_liquidity_score_ticker(t, analyzed_df) for t in ts]))


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

def build_strategy_advisor(analyzed, S, prob_dict, T_expiry, mu, hv, max_loss_pct_limit=100):
    """
    Strategy Advisor 9.3.
    Evalúa estrategias educativas con las opciones disponibles:
    - Call comprado / Put comprado
    - Bull Call Spread / Bear Put Spread
    - Long Straddle / Long Strangle
    - Long Call Butterfly

    Cambios respecto de la versión anterior:
    - Ya no se descartan de entrada las estrategias "contrarias" a la
      predicción dominante (por ej. puts cuando el pronóstico es "Sube").
      Se generan candidatas de todos los tipos disponibles y se las
      ordena por un score cuantitativo; así una estrategia bajista con
      muy buen valor esperado puede aparecer igual, y el usuario puede
      juzgar con datos en vez de con una regla fija.
    - "Prob. éxito est. %" ahora se calcula integrando el payoff de cada
      estrategia contra la misma distribución lognormal usada para las
      probabilidades Sube/Baja/Lateral (antes era una heurística lineal
      por distancia al break-even).
    - Se agrega "Valor esperado" (en la misma unidad que las primas).
    - Se agrega un puntaje de liquidez (volumen + spread compra/venta).
    - El límite de pérdida sobre capital (slider) ahora sí filtra
      estrategias: antes el parámetro se recibía pero no se usaba.
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

    calls = df[df["Tipo"] == "CALL"].sort_values("Strike")
    puts = df[df["Tipo"] == "PUT"].sort_values("Strike")

    def _row_T(row, fallback_T):
        d = clean_num(row.get("Días venc."))
        if not np.isnan(d) and d > 0:
            return d / 365
        return fallback_T

    def add_row(strategy, t1, t2, t3, t4, escenario, ganancia, legs, breakeven, score_base, comment, T_leg=None, venc_str=""):
        T_use = T_leg if T_leg is not None else T_expiry
        payoff, net_cost = strategy_payoff(legs, prices)
        m = strategy_metrics(payoff, net_cost)
        prob_profit, ev = strategy_probability_and_ev(payoff, prices, S, T_use, mu, hv)
        liquidity = liquidity_score_for_legs([t1, t2, t3, t4], analyzed)

        # Acción (Compra/Venta) de cada pata, en el mismo orden que Ticker 1-4.
        # Se arma directamente desde 'legs' (la verdad de la estrategia), no
        # se infiere del nombre -- así no puede quedar desincronizado.
        accion_map = {"buy": "Compra", "sell": "Venta"}
        tickers_row = [t1, t2, t3, t4]
        acciones_partes = []
        for i, leg in enumerate(legs):
            if i < len(tickers_row) and tickers_row[i]:
                qty_leg = leg[4] if len(leg) > 4 else 1
                acciones_partes.append(f"T{i+1}: {accion_map.get(leg[0], leg[0])} x{qty_leg}")
        acciones_str = " · ".join(acciones_partes)

        ev_component = 0.0
        if net_cost:
            ev_ratio = ev / abs(net_cost)
            ev_component = float(np.clip(ev_ratio, -1, 2)) * 10

        strategy_score = (
            score_base * 0.30
            + (prob_profit * 100 if not pd.isna(prob_profit) else 40) * 0.35
            + ev_component
            + liquidity * 0.15
        )
        if m["unlimited_upside"]:
            strategy_score += 8

        over_limit = (not pd.isna(m["loss_pct"])) and (m["loss_pct"] > max_loss_pct_limit)
        if over_limit:
            strategy_score -= 25
        strategy_score = float(np.clip(strategy_score, 0, 130))

        rows.append({
            "Ranking": "",
            "Estrategia": strategy,
            "Ticker 1": t1,
            "Ticker 2": t2,
            "Ticker 3": t3,
            "Ticker 4": t4,
            "Acción por pata": acciones_str,
            "Vencimiento": venc_str,
            "Días venc.": int(round(T_use * 365)),
            "Escenario": escenario,
            "Ganancia": ganancia,
            "Costo neto": net_cost,
            "Pérdida máx.": m["max_loss"],
            "% pérdida/capital": m["loss_pct"],
            "Break-even": breakeven,
            "Prob. éxito est. %": prob_profit * 100 if not pd.isna(prob_profit) else np.nan,
            "Valor esperado": ev,
            "Liquidez": liquidity,
            "Dentro del límite de pérdida": not over_limit,
            "Score estrategia": strategy_score,
            "Comentario": comment,
            "Legs": legs,
        })

    # Long calls: upside ilimitado
    if not calls.empty:
        candidate_calls = calls[(calls["Strike"] >= S * 0.92) & (calls["Strike"] <= S * 1.18)].sort_values("Score Busa", ascending=False).head(10)
        for _, c in candidate_calls.iterrows():
            K = float(c["Strike"]); p = float(c["prima_ref"])
            legs = [("buy", "call", K, p, 1)]
            breakeven = K + p
            add_row("Call comprado", c["Ticker"], "", "", "", "Alcista fuerte", "Ilimitada teórica", legs, breakeven, float(c["Score Busa"]), "Mayor potencial alcista. Riesgo limitado a prima.", T_leg=_row_T(c, T_expiry), venc_str=str(c.get("Vencimiento", "")))

    # Long puts
    if not puts.empty:
        candidate_puts = puts[(puts["Strike"] >= S * 0.82) & (puts["Strike"] <= S * 1.08)].sort_values("Score Busa", ascending=False).head(10)
        for _, p_row in candidate_puts.iterrows():
            K = float(p_row["Strike"]); p = float(p_row["prima_ref"])
            legs = [("buy", "put", K, p, 1)]
            breakeven = K - p
            add_row("Put comprado", p_row["Ticker"], "", "", "", "Bajista fuerte", "Alta, limitada por subyacente a cero", legs, breakeven, float(p_row["Score Busa"]), "Potencial bajista con riesgo limitado a prima.", T_leg=_row_T(p_row, T_expiry), venc_str=str(p_row.get("Vencimiento", "")))

    # Bull call spreads (sólo entre opciones del mismo vencimiento)
    if len(calls) >= 2:
        base_calls = calls[(calls["Strike"] >= S * 0.92) & (calls["Strike"] <= S * 1.10)].sort_values("Score Busa", ascending=False).head(6)
        for _, buy in base_calls.iterrows():
            higher = calls[calls["Strike"] > buy["Strike"]].head(5)
            for _, sell in higher.iterrows():
                venc_buy, venc_sell = str(buy.get("Vencimiento", "")), str(sell.get("Vencimiento", ""))
                if venc_buy and venc_sell and venc_buy != venc_sell:
                    continue  # no mezclar vencimientos distintos en un mismo spread
                p_buy = float(buy["prima_ref"]); p_sell = float(sell["prima_ref"]); net = p_buy - p_sell
                if net <= 0: continue
                legs = [("buy", "call", float(buy["Strike"]), p_buy, 1), ("sell", "call", float(sell["Strike"]), p_sell, 1)]
                breakeven = float(buy["Strike"]) + net
                score_base = np.nanmean([buy["Score Busa"], sell["Score Busa"]])
                add_row("Bull Call Spread", buy["Ticker"], sell["Ticker"], "", "", "Alcista moderado", "Limitada", legs, breakeven, score_base, "Menor costo y menor riesgo que call comprado.", T_leg=_row_T(buy, T_expiry), venc_str=venc_buy or venc_sell)

    # Bear put spreads (sólo entre opciones del mismo vencimiento)
    if len(puts) >= 2:
        base_puts = puts[(puts["Strike"] >= S * 0.90) & (puts["Strike"] <= S * 1.08)].sort_values("Score Busa", ascending=False).head(6)
        for _, buy in base_puts.iterrows():
            lower = puts[puts["Strike"] < buy["Strike"]].tail(5)
            for _, sell in lower.iterrows():
                venc_buy, venc_sell = str(buy.get("Vencimiento", "")), str(sell.get("Vencimiento", ""))
                if venc_buy and venc_sell and venc_buy != venc_sell:
                    continue
                p_buy = float(buy["prima_ref"]); p_sell = float(sell["prima_ref"]); net = p_buy - p_sell
                if net <= 0: continue
                legs = [("buy", "put", float(buy["Strike"]), p_buy, 1), ("sell", "put", float(sell["Strike"]), p_sell, 1)]
                breakeven = float(buy["Strike"]) - net
                score_base = np.nanmean([buy["Score Busa"], sell["Score Busa"]])
                add_row("Bear Put Spread", buy["Ticker"], sell["Ticker"], "", "", "Bajista moderado", "Limitada", legs, breakeven, score_base, "Menor costo y menor riesgo que put comprado.", T_leg=_row_T(buy, T_expiry), venc_str=venc_buy or venc_sell)

    # Call Ratio Backspread (sólo entre opciones del mismo vencimiento):
    # vender 1 call de strike más cercano y comprar 2 de strike más lejano.
    # No es de las 6 estrategias "de manual" de arriba, pero es una
    # combinación real y conocida que estructuralmente da pérdida acotada
    # en la zona media + ganancia SIN TECHO si el papel sube fuerte. Se
    # verifica el resultado calculado, no se asume por construcción.
    if len(calls) >= 2:
        base_calls_bs = calls[(calls["Strike"] >= S * 0.90) & (calls["Strike"] <= S * 1.05)].sort_values("Score Busa", ascending=False).head(6)
        for _, sell in base_calls_bs.iterrows():
            higher = calls[calls["Strike"] > sell["Strike"]].head(5)
            for _, buy in higher.iterrows():
                venc_sell, venc_buy = str(sell.get("Vencimiento", "")), str(buy.get("Vencimiento", ""))
                if venc_sell and venc_buy and venc_sell != venc_buy:
                    continue
                p_sell = float(sell["prima_ref"]); p_buy = float(buy["prima_ref"])
                legs = [("sell", "call", float(sell["Strike"]), p_sell, 1), ("buy", "call", float(buy["Strike"]), p_buy, 2)]
                payoff_bs, net_cost_bs = strategy_payoff(legs, prices)
                m_bs = strategy_metrics(payoff_bs, net_cost_bs)
                if not m_bs["unlimited_upside"]:
                    continue  # nos interesan sólo los que sí confirman ganancia ilimitada
                score_base = np.nanmean([sell["Score Busa"], buy["Score Busa"]])
                add_row(
                    "Call Ratio Backspread (1:2)", sell["Ticker"], buy["Ticker"], "", "",
                    "Alcista fuerte", "Ilimitada al alza",
                    legs, np.nan, score_base,
                    "No estándar: vendés 1 call más cercana y comprás 2 más lejanas. Pérdida acotada si el papel sube poco o queda quieto; sin techo si sube fuerte.",
                    T_leg=_row_T(sell, T_expiry), venc_str=venc_sell or venc_buy,
                )

    # Put Ratio Backspread (mismo vencimiento): vender 1 put de strike más
    # alto y comprar 2 de strike más bajo. Pérdida acotada en la zona media
    # + ganancia grande (acotada por el subyacente yendo a cero, pero mucho
    # mayor que la pérdida posible) si el papel cae fuerte.
    if len(puts) >= 2:
        base_puts_bs = puts[(puts["Strike"] >= S * 0.95) & (puts["Strike"] <= S * 1.10)].sort_values("Score Busa", ascending=False).head(6)
        for _, sell in base_puts_bs.iterrows():
            lower = puts[puts["Strike"] < sell["Strike"]].tail(5)
            for _, buy in lower.iterrows():
                venc_sell, venc_buy = str(sell.get("Vencimiento", "")), str(buy.get("Vencimiento", ""))
                if venc_sell and venc_buy and venc_sell != venc_buy:
                    continue
                p_sell = float(sell["prima_ref"]); p_buy = float(buy["prima_ref"])
                legs = [("sell", "put", float(sell["Strike"]), p_sell, 1), ("buy", "put", float(buy["Strike"]), p_buy, 2)]
                payoff_bs2, net_cost_bs2 = strategy_payoff(legs, prices)
                # Ganancia fuerte a la baja: el extremo izquierdo del rango
                # de precios simulado debe rendir mucho más que la pérdida
                # máxima en la zona media.
                m_bs2 = strategy_metrics(payoff_bs2, net_cost_bs2)
                gana_fuerte_abajo = payoff_bs2[0] > payoff_bs2[1] and payoff_bs2[0] > 2 * abs(m_bs2["max_loss"])
                if not gana_fuerte_abajo:
                    continue
                score_base = np.nanmean([sell["Score Busa"], buy["Score Busa"]])
                add_row(
                    "Put Ratio Backspread (1:2)", sell["Ticker"], buy["Ticker"], "", "",
                    "Bajista fuerte", "Alta a la baja",
                    legs, np.nan, score_base,
                    "No estándar: vendés 1 put más cercana y comprás 2 más lejanas. Pérdida acotada si el papel baja poco o queda quieto; ganancia grande si cae fuerte.",
                    T_leg=_row_T(sell, T_expiry), venc_str=venc_sell or venc_buy,
                )

    # Straddle / Strangle long for movement (sólo entre opciones del mismo vencimiento)
    if len(calls) >= 1 and len(puts) >= 1:
        near_calls = calls.iloc[(calls["Strike"] - S).abs().argsort()[:4]]
        near_puts = puts.iloc[(puts["Strike"] - S).abs().argsort()[:4]]
        for _, c in near_calls.iterrows():
            for _, p_row in near_puts.iterrows():
                venc_c, venc_p = str(c.get("Vencimiento", "")), str(p_row.get("Vencimiento", ""))
                if venc_c and venc_p and venc_c != venc_p:
                    continue
                pc = float(c["prima_ref"]); pp = float(p_row["prima_ref"])
                legs = [("buy", "call", float(c["Strike"]), pc, 1), ("buy", "put", float(p_row["Strike"]), pp, 1)]
                if abs(float(c["Strike"]) - float(p_row["Strike"])) < 1e-9:
                    strat = "Long Straddle"
                else:
                    strat = "Long Strangle"
                bkeven = np.nan
                score_base = np.nanmean([c["Score Busa"], p_row["Score Busa"]])
                add_row(strat, c["Ticker"], p_row["Ticker"], "", "", "Movimiento fuerte", "Ilimitada al alza / alta a la baja", legs, bkeven, score_base, "Apuesta a movimiento fuerte. Riesgo limitado a primas.", T_leg=_row_T(c, T_expiry), venc_str=venc_c or venc_p)

    # Butterfly with calls: lateral / target (mismo vencimiento en las 3 patas por construcción, ya que se arma dentro de 'calls')
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
            venc1, venc2, venc3 = str(r1.get("Vencimiento", "")), str(r2.get("Vencimiento", "")), str(r3.get("Vencimiento", ""))
            vencs_conocidos = [v for v in [venc1, venc2, venc3] if v]
            if len(set(vencs_conocidos)) > 1:
                continue  # las 3 patas deben compartir vencimiento
            legs = [
                ("buy", "call", float(k1), float(r1["prima_ref"]), 1),
                ("sell", "call", float(k2), float(r2["prima_ref"]), 2),
                ("buy", "call", float(k3), float(r3["prima_ref"]), 1),
            ]
            score_base = np.nanmean([r1["Score Busa"], r2["Score Busa"], r3["Score Busa"]])
            add_row("Long Call Butterfly", r1["Ticker"], r2["Ticker"], r3["Ticker"], "", "Lateral / objetivo cercano", "Limitada", legs, np.nan, score_base, "Riesgo definido. Busca cierre cerca del strike central.", T_leg=_row_T(r1, T_expiry), venc_str=venc1 or venc2 or venc3)
            break

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Filtra por el límite de pérdida elegido; si el filtro deja todo afuera,
    # se conserva el listado completo (con el puntaje ya penalizado) para no
    # dejar la pantalla vacía.
    within_limit = out[out["Dentro del límite de pérdida"]]
    if not within_limit.empty:
        out = within_limit

    out = out.sort_values("Score estrategia", ascending=False).reset_index(drop=True)
    out["Ranking"] = np.arange(1, len(out)+1)
    return out


# =========================
# Defaults robustos para móvil/cloud
# =========================
if "activo_select" not in st.session_state or st.session_state.get("activo_select") not in ["GGAL", "YPF"]:
    st.session_state["activo_select"] = "GGAL"
if "prima_mode_select" not in st.session_state or st.session_state.get("prima_mode_select") not in ["Promedio compra/venta", "Venta", "Compra", "Último"]:
    st.session_state["prima_mode_select"] = "Promedio compra/venta"

# =========================
# Sidebar
# =========================
with st.sidebar:
    usage = load_usage()
    st.header("Actualizar")
    st.caption(market_status_text())
    st.metric("Consultas mes", f"{usage.get('calls', 0):,} / {LIMIT:,}")
    st.caption("En Streamlit Cloud el contador puede reiniciarse tras redeploy/reboot.")
    st.progress(min(1, usage.get("calls", 0) / LIMIT))
    if usage.get("last_update"):
        st.caption(f"Última API: {usage['last_update']}")

    activo = st.selectbox("Activo", ["GGAL", "YPF"], key="activo_select")
    mode = st.selectbox("Prima usada", ["Promedio compra/venta", "Venta", "Compra", "Último"], key="prima_mode_select")

    with st.expander("Parámetros", expanded=False):
        period = st.selectbox("Histórico", ["6mo", "1y", "2y", "5y"], index=2)
        lookback = st.number_input("VH ruedas", 20, 252, 60, 5)
        horizon = st.number_input("Horizonte", 1, 120, 20)
        lateral = st.number_input("Lateral +/- %", 0.5, 30.0, 5.0, .5) / 100
        r = st.number_input("Tasa caución %", 0.0, 200.0, 20.2, .1) / 100
        days = st.number_input("Días vencimiento (respaldo)", 1, 365, 52,
                                help="Ya no es el valor principal: cada opción calcula su propio vencimiento automáticamente a partir del ticker (tercer viernes del mes que indican sus 2 últimas letras). Esto solo se usa si un ticker no se puede interpretar.")

    with st.expander("Modelo de pronóstico", expanded=False):
        st.caption("El retorno medio histórico es un estimador ruidoso de la tendencia futura. Estos controles moderan ese ruido.")
        drift_shrink = st.slider("Sensibilidad a la tendencia histórica", 0.0, 1.0, 0.35, 0.05,
                                  help="0 = ignora la tendencia histórica reciente (pronóstico centrado). 1 = la usa completa, como antes (más ruidoso).")
        use_tilt = st.checkbox("Sumar sesgo técnico (RSI/momentum/EMAs)", value=True,
                                help="Ajuste chico y acotado, no reemplaza al modelo estadístico.")
        tilt_strength = st.slider("Fuerza del sesgo técnico", 0.0, 2.0, 1.0, 0.1) if use_tilt else 0.0

    with st.expander("Aprendizaje (avanzado)", expanded=False):
        st.caption("Cuántas señales evaluadas recientes considera el Learning, y qué tan rápido reacciona.")
        learning_window = st.number_input("Ventana de señales evaluadas", 10, 100, 40, 5)
        learning_prior_strength = st.slider("Peso del prior (más alto = más lento para reaccionar)", 1, 20, 6, 1)

    if st.button("🔄 Actualizar mercado (IOL)"):
        try:
            client = IOLClient.from_config()

            raw = client.get_options(TICKERS[activo]["iol"])
            quote_raw = client.get_quote(TICKERS[activo]["iol"])
            spot_iol = extract_iol_quote_price(quote_raw)

            st.session_state["raw_iol"] = raw
            st.session_state["quote_iol"] = quote_raw
            st.session_state["options_df"] = normalize_options(raw)
            # Vol. Compra/Vol. Venta del PAPEL (no de las opciones): misma
            # función genérica que ya se usa para las puntas de cada opción,
            # aplicada acá a la cotización del subyacente.
            st.session_state["underlying_book"] = extract_option_quote_fields(quote_raw)

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
                prob_tmp = prob_data(h_tmp, int(horizon), lateral, int(lookback), drift_shrink, use_tilt, tilt_strength)
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
prob = prob_data(h, int(horizon), lateral, int(lookback), drift_shrink, use_tilt, tilt_strength)
prob = apply_learning_to_probabilities(prob, activo, int(learning_window), int(learning_prior_strength))
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
tabs = st.tabs(["Dashboard", "Opciones", "Probabilidades", "Busa AI", "Advisor", "Favoritos"])

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

    underlying_book = st.session_state.get("underlying_book", {})
    if underlying_book and (not np.isnan(clean_num(underlying_book.get("Vol. Compra"))) or not np.isnan(clean_num(underlying_book.get("Vol. Venta")))):
        st.markdown(f"##### Volumen de puntas — {activo} (papel, vía IOL)")
        b1, b2, b3 = st.columns(3)
        b1.metric("Vol. Compra", "" if np.isnan(clean_num(underlying_book.get("Vol. Compra"))) else f"{underlying_book['Vol. Compra']:,.0f}")
        b2.metric("Vol. Venta", "" if np.isnan(clean_num(underlying_book.get("Vol. Venta"))) else f"{underlying_book['Vol. Venta']:,.0f}")
        b3.metric("Volumen operado", "" if np.isnan(clean_num(underlying_book.get("Volumen"))) else f"{underlying_book['Volumen']:,.0f}")
        st.caption("Vol. Compra/Venta: suma de la cantidad ofrecida en todos los niveles de puntas que devuelve IOL para el papel (no de las opciones). Volumen operado: nominales operados en el día.")

    with st.expander("🔎 Cruzar con otra fuente (BYMA / PyOBD, experimental)", expanded=False):
        st.caption("Segunda fuente independiente de IOL para el papel (no cubre opciones). Usa datos abiertos de BYMA sin cuenta de broker. No pude validar el formato exacto de respuesta desde acá -- si algo no aparece, mandame lo que se ve en 'Ver respuesta cruda' para ajustar el parseo.")
        if st.button("Traer cotización BYMA (PyOBD)"):
            byma_raw = get_secondary_quote_byma(TICKERS[activo]["local"])
            st.session_state["byma_secondary_raw"] = byma_raw
            st.rerun()
        byma_raw = st.session_state.get("byma_secondary_raw")
        if byma_raw is None:
            st.info("Todavía no trajiste datos de BYMA, o la librería PyOBD no está disponible/instalada.")
        else:
            byma_fields = parse_byma_secondary_fields(byma_raw)
            cb1, cb2, cb3 = st.columns(3)
            cb1.metric("Compra (BYMA)", "" if np.isnan(clean_num(byma_fields.get("Compra (BYMA)"))) else f"{byma_fields['Compra (BYMA)']:,.2f}")
            cb2.metric("Venta (BYMA)", "" if np.isnan(clean_num(byma_fields.get("Venta (BYMA)"))) else f"{byma_fields['Venta (BYMA)']:,.2f}")
            cb3.metric("Último (BYMA)", "" if np.isnan(clean_num(byma_fields.get("Último (BYMA)"))) else f"{byma_fields['Último (BYMA)']:,.2f}")
            cb4, cb5, cb6 = st.columns(3)
            cb4.metric("Vol. Compra (BYMA)", "" if np.isnan(clean_num(byma_fields.get("Vol. Compra (BYMA)"))) else f"{byma_fields['Vol. Compra (BYMA)']:,.0f}")
            cb5.metric("Vol. Venta (BYMA)", "" if np.isnan(clean_num(byma_fields.get("Vol. Venta (BYMA)"))) else f"{byma_fields['Vol. Venta (BYMA)']:,.0f}")
            cb6.metric("Volumen (BYMA)", "" if np.isnan(clean_num(byma_fields.get("Volumen (BYMA)"))) else f"{byma_fields['Volumen (BYMA)']:,.0f}")
            with st.expander("Ver respuesta cruda de PyOBD"):
                st.json(byma_raw)

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
    st.subheader("Probabilidades y Análisis Técnico")
    st.caption("Pronóstico Busa AI (modelo estadístico + Learning) y análisis técnico clásico — RSI, MACD, Bandas de Bollinger y EMAs — para GGAL e YPF en BYMA.")

    ticker_tabs = st.tabs(["GGAL", "YPF"])
    with ticker_tabs[0]:
        render_technical_panel("GGAL", activo, period, horizon, lateral, lookback,
                                drift_shrink, use_tilt, tilt_strength, learning_window, learning_prior_strength)
    with ticker_tabs[1]:
        render_technical_panel("YPF", activo, period, horizon, lateral, lookback,
                                drift_shrink, use_tilt, tilt_strength, learning_window, learning_prior_strength)


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
    generado_ai = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ultimo_dato_ai = pd.to_datetime(close.index[-1]).strftime("%d/%m/%Y")
    st.caption(f"🕒 Predicción generada: {generado_ai} hs. | Datos de precio hasta: {ultimo_dato_ai}")

    st.markdown("### Probabilidades")
    p1, p2, p3 = st.columns(3)
    p1.metric("Sube", f"{prob['Sube']:.1%}")
    p2.metric("Baja", f"{prob['Baja']:.1%}")
    p3.metric("Lateral", f"{prob['Lateral']:.1%}")

    st.markdown("### Aprendizaje visible")
    st.caption("Evaluada=0 significa pendiente. Cuando se evalúa, Resultado y Acierto muestran si predijo bien o mal.")
    l1, l2, l3 = st.columns(3)
    l1.metric("Learning factor (clase dominante)", f"{prob.get('Learning factor', 1.0):.2f}")
    l2.metric("Señales evaluadas (activo)", n_eval)
    l3.metric("Accuracy histórico (activo)", "" if pd.isna(acc) else f"{acc*100:.1f}%")

    if prob.get("Learning factor", 1.0) == 1.0:
        st.info("El modelo todavía está cerca del neutral: necesita más señales evaluadas o el accuracy no justifica ajustar.")
    elif prob.get("Learning factor", 1.0) > 1.0:
        st.success("El modelo está reforzando la predicción dominante porque el historial viene acompañando.")
    else:
        st.warning("El modelo está moderando la predicción dominante porque el historial viene fallando.")

    st.markdown("#### Detalle por clase (Sube / Baja / Lateral)")
    st.caption("Cada clase se ajusta con su propio historial de aciertos (Bayesiano, no un umbral fijo). 'n' = señales evaluadas de esa clase específica; con pocas señales el factor queda cerca de 1.0.")
    learning_stats = prob.get("Learning stats")
    if learning_stats:
        st.dataframe(pd.DataFrame([
            {
                "Clase": c,
                "Factor": learning_stats[c]["factor"],
                "n evaluadas": learning_stats[c]["n"],
                "Accuracy cruda %": np.nan if pd.isna(learning_stats[c]["acc_raw"]) else learning_stats[c]["acc_raw"] * 100,
                "Accuracy suavizada %": learning_stats[c]["acc_post"] * 100,
            }
            for c in ["Sube", "Baja", "Lateral"]
        ]), use_container_width=True)
    else:
        st.info("Todavía no hay suficientes señales evaluadas por clase.")

    quality = forecast_quality_summary(activo)
    st.markdown("#### Calidad de calibración")
    if quality is None:
        st.info("Todavía no hay señales evaluadas para medir calibración.")
    else:
        q1, q2, q3 = st.columns(3)
        q1.metric("Señales usadas", quality["n"])
        q2.metric("Accuracy", "" if pd.isna(quality["accuracy"]) else f"{quality['accuracy']*100:.1f}%")
        q3.metric("Brier score", "" if pd.isna(quality["brier"]) else f"{quality['brier']:.3f}")
        st.caption("Brier score: 0 = probabilidades perfectamente calibradas, 2 = lo peor posible. Es más exigente que el accuracy porque también penaliza estar 'demasiado seguro' cuando se falla.")

    st.markdown("### Estrategias sugeridas por Busa AI")
    st.dataframe(pd.DataFrame(option_strategy_suggestions(pred, confidence)), use_container_width=True)

    st.markdown("### Por qué Busa AI interpreta esto")
    reasons = busa_ai_reason_cards(prob, hv, S, h)
    for r_reason in reasons[:6]:
        st.write(f"✔ {r_reason}")

    with st.expander("Cómo se arma el pronóstico (drift y volatilidad)", expanded=False):
        st.caption("Desglose del modelo estadístico antes de aplicar el ajuste de Learning.")
        st.write(f"**Retorno histórico anualizado (crudo):** {prob.get('Mu hist', 0)*100:.1f}%")
        st.write(f"**Retorno usado en el modelo (con shrinkage + sesgo técnico):** {prob.get('Mu ajustada', 0)*100:.1f}%")
        st.write(f"**Sesgo técnico aplicado:** {prob.get('Tilt técnico', 0)*100:.2f} puntos anualizados")
        st.write(f"**Volatilidad simple:** {prob.get('VH simple', hv)*100:.1f}% | **Volatilidad EWMA:** {prob.get('VH EWMA', hv)*100:.1f}% | **Volatilidad usada:** {hv*100:.1f}%")
        st.caption("El shrinkage evita que una racha corta de suba/baja se extrapole como si fuera a repetirse. Ajustable en la barra lateral, sección 'Modelo de pronóstico'.")

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

    st.markdown("### Backup del Learning")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "⬇️ Descargar historial Learning CSV",
            data=predictions_csv_bytes(),
            file_name="busaoptions_learning_backup.csv",
            mime="text/csv",
        )
    with b2:
        uploaded_learning = st.file_uploader("Restaurar Learning CSV", type=["csv"], key="restore_learning_csv")
        if uploaded_learning is not None and st.button("Restaurar historial"):
            if restore_predictions_from_upload(uploaded_learning):
                st.success("Historial restaurado.")
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
    st.subheader("Strategy Advisor 9.3")
    st.caption("Motor educativo: long call/put, spreads, straddle, strangle y butterfly. Rankea por probabilidad de éxito y valor esperado calculados con el mismo modelo de pronóstico, más liquidez.")

    if analyzed.empty:
        st.warning("Primero cargá opciones con Actualizar mercado (IOL).")
    else:
        pred_adv = dominant_prediction(prob)
        confidence_adv = busa_ai_confidence_label(prob)
        st.info(f"Escenario detectado: **{pred_adv}** | Confianza: **{confidence_adv}**")
        st.caption("El ranking evalúa todas las estrategias disponibles (no sólo las del escenario detectado); una estrategia contraria puede aparecer arriba si su probabilidad/valor esperado son mejores.")

        max_loss_pct = st.slider("Límite máximo de pérdida sobre capital (%)", 10, 200, 100, 5)
        advisor = build_strategy_advisor(analyzed, S, prob, T, prob.get("Mu ajustada", 0.0), hv, max_loss_pct)

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

            c5, c6 = st.columns(2)
            c5.metric("Valor esperado", f"{best['Valor esperado']:.2f}")
            c6.metric("Liquidez", f"{best['Liquidez']:.0f}/100")

            st.write(f"**Qué comprar/vender:** {best['Acción por pata']}")
            st.write(f"**Comentario:** {best['Comentario']}")
            st.write(f"**Ganancia:** {best['Ganancia']}")
            st.write(f"**Pérdida máxima estimada:** {best['Pérdida máx.']:.2f}")
            if not pd.isna(best["% pérdida/capital"]):
                st.write(f"**% pérdida sobre capital arriesgado:** {best['% pérdida/capital']:.1f}%")
            if not pd.isna(best["Break-even"]):
                st.write(f"**Break-even:** {best['Break-even']:.2f}")

            st.markdown("### Ranking de estrategias")
            st.caption("Columna 'Acción por pata': indica Compra/Venta para cada Ticker 1-4, en el mismo orden. Se arma directo de la estrategia real, no hay que adivinarlo por el nombre.")
            display_cols = [c for c in advisor.columns if c != "Legs"]
            st.dataframe(advisor[display_cols].head(20), use_container_width=True)

            # Payoff: por default muestra la #1, pero se puede elegir
            # cualquier fila del ranking (antes sólo se graficaba la #1).
            st.markdown("### Gráfico de payoff")
            top_n = advisor.head(20).reset_index(drop=True)
            opciones_chart = [
                f"{int(r['Ranking'])} - {r['Estrategia']} ({r['Ticker 1']}" + (f" / {r['Ticker 2']}" if r["Ticker 2"] else "") + ")"
                for _, r in top_n.iterrows()
            ]
            idx_chart = st.selectbox("Elegí qué estrategia del ranking graficar", range(len(opciones_chart)), format_func=lambda i: opciones_chart[i])
            elegida = top_n.iloc[idx_chart]

            prices_adv = np.linspace(S * 0.70, S * 1.35, 300)
            legs_adv = elegida.get("Legs", [])

            if legs_adv:
                payoff_adv, net_cost_adv = strategy_payoff(legs_adv, prices_adv)
                fig_adv = go.Figure()
                fig_adv.add_trace(go.Scatter(x=prices_adv, y=payoff_adv, name="Payoff", mode="lines"))
                fig_adv.add_hline(y=0, line_dash="dash")
                fig_adv.add_vline(x=S, line_dash="dot", annotation_text="Precio actual")
                fig_adv.update_layout(template="plotly_dark", height=460, xaxis_title="Precio al vencimiento", yaxis_title="Resultado", hovermode="x unified")
                st.plotly_chart(fig_adv, use_container_width=True)
                st.caption(f"Patas: {elegida['Acción por pata']}")

            st.caption("Herramienta educativa. No constituye recomendación financiera personalizada.")


with tabs[5]:
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

    st.divider()
    st.markdown("### Análisis fundamental y ADR (exterior)")
    st.caption("Datos fundamentales y cotización del ADR (la acción cotizando afuera, en USD: NASDAQ para GGAL, NYSE para YPF). Se actualiza una vez por día automáticamente; el botón fuerza un refresco al toque.")

    fund_ticker_choice = st.selectbox("Activo para el análisis fundamental", ["GGAL", "YPF"], key="fund_ticker_choice")

    if st.button("🔄 Actualizar análisis fundamental"):
        get_fundamentals.clear()
        st.session_state["fundamentals_updated"] = {}

    adr_ticker = TICKERS[fund_ticker_choice]["adr"]
    info_fund = get_fundamentals(adr_ticker)
    fund = extract_fundamentals(info_fund)

    if not fund or fund.get("Precio ADR (USD)") is None:
        st.warning("No pude traer datos fundamentales de Yahoo Finance para este ticker en este momento. Puede ser un corte temporal del proveedor -- probá 'Actualizar' en un rato.")
    else:
        ultima_act = st.session_state.get("fundamentals_updated", {}).get(fund_ticker_choice)
        if ultima_act is None:
            ultima_act = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            st.session_state.setdefault("fundamentals_updated", {})[fund_ticker_choice] = ultima_act
        st.caption(f"🕒 Datos al: {ultima_act} hs. (caché de hasta 12hs; usá el botón para forzar un refresco)")

        st.markdown(f"##### {fund_ticker_choice} ADR ({adr_ticker}) — cómo está la acción en el exterior")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Precio ADR (USD)", "" if fund["Precio ADR (USD)"] is None else f"US$ {fund['Precio ADR (USD)']:.2f}")
        p2.metric("Variación día", "" if fund["Variación día %"] is None else f"{fund['Variación día %']:.2f}%")
        p3.metric("Máx. 52 sem.", "" if fund["Máx. 52 sem."] is None else f"US$ {fund['Máx. 52 sem.']:.2f}")
        p4.metric("Mín. 52 sem.", "" if fund["Mín. 52 sem."] is None else f"US$ {fund['Mín. 52 sem.']:.2f}")

        st.markdown("##### Fundamentales")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Market Cap (USD)", "" if fund["Market Cap (USD)"] is None else f"{fund['Market Cap (USD)']/1e9:.2f}B")
        f2.metric("P/E (trailing)", "" if fund["P/E (trailing)"] is None else f"{fund['P/E (trailing)']:.1f}")
        f3.metric("P/E (forward)", "" if fund["P/E (forward)"] is None else f"{fund['P/E (forward)']:.1f}")
        f4.metric("P/B", "" if fund["P/B"] is None else f"{fund['P/B']:.2f}")

        f5, f6, f7, f8 = st.columns(4)
        f5.metric("Dividend Yield", "" if fund["Dividend Yield %"] is None else f"{fund['Dividend Yield %']*100:.2f}%")
        f6.metric("ROE", "" if fund["ROE %"] is None else f"{fund['ROE %']*100:.1f}%")
        f7.metric("Margen neto", "" if fund["Margen neto %"] is None else f"{fund['Margen neto %']*100:.1f}%")
        f8.metric("Beta", "" if fund["Beta"] is None else f"{fund['Beta']:.2f}")

        f9, f10 = st.columns(2)
        f9.metric("EPS (TTM, USD)", "" if fund["EPS (TTM, USD)"] is None else f"{fund['EPS (TTM, USD)']:.2f}")
        f10.metric("Ingresos TTM (USD)", "" if fund["Ingresos TTM (USD)"] is None else f"{fund['Ingresos TTM (USD)']/1e9:.2f}B")

        st.markdown("##### Analistas (Wall Street)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Precio objetivo prom.", "" if fund["Precio objetivo analistas (USD)"] is None else f"US$ {fund['Precio objetivo analistas (USD)']:.2f}")
        a2.metric("Recomendación", fund["Recomendación analistas"] or "")
        a3.metric("Cantidad de analistas", fund["Cantidad analistas"] or "")
        if fund["Precio objetivo analistas (USD)"] is not None and fund["Precio ADR (USD)"] is not None and fund["Precio ADR (USD)"] > 0:
            upside = (fund["Precio objetivo analistas (USD)"] / fund["Precio ADR (USD)"] - 1) * 100
            st.caption(f"El precio objetivo promedio de los analistas implica un {'potencial alcista' if upside >= 0 else 'potencial bajista'} de {abs(upside):.1f}% respecto del precio actual del ADR.")

        st.caption("Fuente: Yahoo Finance (ADR). Los fundamentales de la especie local en BYMA no suelen estar disponibles ahí, por eso se usa el ADR como referencia. Esto es información, no una recomendación de inversión.")

    st.divider()
    st.markdown("### Mi cartera (IOL)")
    st.caption("Trae tus posiciones de opciones de GGAL e YPF desde IOL y sugiere Vender / Mantener / Vigilar según el pronóstico Busa AI y el veredicto técnico vigentes.")
    st.caption("El vencimiento de cada posición se calcula solo, a partir del ticker. Si alguno no se puede interpretar, cae al respaldo configurado en 'Parámetros' de la barra lateral.")

    donchian_window_cartera = st.slider(
        "Ventana para Máx./Mín. de la prima (ruedas)", 5, 30, 10, 1,
        help="Canal de Donchian: marca el máximo y el mínimo efectivamente tocados por la prima en las últimas N ruedas, más una media móvil simple de esa misma ventana.",
    )

    if st.button("📂 Traer mi cartera de IOL"):
        try:
            client = IOLClient.from_config()
            raw_port = client.get_portfolio("argentina")
            st.session_state["portfolio_raw"] = raw_port
            st.session_state["portfolio_df"] = normalize_portfolio(raw_port)
            st.rerun()
        except (IOLAuthError, IOLApiError) as e:
            st.error(f"Error IOL: {e}")
        except Exception as e:
            st.error(f"No pude traer la cartera: {e}")

    port_df = st.session_state.get("portfolio_df", pd.DataFrame())
    if port_df.empty:
        st.info("Todavía no trajiste tu cartera, o no hay posiciones. Tocá 'Traer mi cartera de IOL'.")
    else:
        opciones_port = port_df[port_df.apply(is_option_position, axis=1)].copy()
        if opciones_port.empty:
            st.warning("Traje tu cartera pero no encontré posiciones de opciones de GGAL/YPF reconocibles (series GFGC/GFGV/YPFC/YPFV). Revisá 'Debug IOL' más abajo para ver la respuesta cruda: si tus tickers vienen con otro formato, avisame para ajustar el parseo.")
            st.dataframe(port_df, use_container_width=True)
        else:
            resultados = []
            detalles_extra = {}

            client_pos = None
            try:
                client_pos = IOLClient.from_config()
            except Exception:
                client_pos = None

            for _, pos in opciones_port.iterrows():
                ticker = pos["Ticker"]
                subyacente = underlying_for_option(ticker)
                if subyacente is None:
                    continue
                typ = infer_tipo(ticker)

                h_pos = get_hist(TICKERS[subyacente]["local"], period)
                if h_pos.empty:
                    continue
                prob_pos = prob_data(h_pos, int(horizon), lateral, int(lookback), drift_shrink, use_tilt, tilt_strength)
                prob_pos = apply_learning_to_probabilities(prob_pos, subyacente, int(learning_window), int(learning_prior_strength))
                S_pos = float(prob_pos["S"])
                hv_pos = prob_pos["VH"]
                close_pos = h_pos["Close"].dropna()

                rsi_pos = compute_rsi(close_pos)
                macd_pos, macd_signal_pos, macd_hist_pos = compute_macd(close_pos)
                sma20_pos, bb_up_pos, bb_dn_pos = compute_bollinger(close_pos)
                ema50_pos = close_pos.ewm(span=50, adjust=False).mean()
                adx_pos, plus_di_pos, minus_di_pos = compute_adx(h_pos["High"], h_pos["Low"], close_pos)
                volumen_pos_series = h_pos["Volume"] if "Volume" in h_pos.columns else None
                veredicto_pos, _, _, _, _ = technical_verdict(
                    rsi_pos, macd_pos, macd_signal_pos, close_pos, sma20_pos, ema50_pos,
                    volumen_pos_series, adx_pos, plus_di_pos, minus_di_pos,
                )

                m = re.search(r"(\d{3,6})", ticker)
                strike = clean_num(m.group(1)) if m else np.nan

                # Vencimiento automático por ticker; si no se puede interpretar,
                # cae al respaldo manual.
                fecha_venc_pos, dias_venc_pos, fuente_venc_pos = parse_option_expiry(ticker)
                if fecha_venc_pos is None:
                    dias_venc_pos = days
                    fuente_venc_pos = "manual"
                T_pos = max(dias_venc_pos, 1) / 365

                # Cotización en vivo de ESTA opción puntual (más precisa que el
                # 'UltimoPrecio' de la cartera, que puede estar desactualizado).
                compra_op = venta_op = vol_compra_op = vol_venta_op = np.nan
                if client_pos is not None:
                    try:
                        q_pos = client_pos.get_quote(ticker)
                        campos_q = extract_option_quote_fields(q_pos)
                        compra_op = campos_q.get("Compra", np.nan)
                        venta_op = campos_q.get("Venta", np.nan)
                        vol_compra_op = campos_q.get("Vol. Compra", np.nan)
                        vol_venta_op = campos_q.get("Vol. Venta", np.nan)
                    except Exception:
                        pass

                if not np.isnan(compra_op) and not np.isnan(venta_op):
                    prima_actual = (compra_op + venta_op) / 2
                    fuente_precio_pos = "cotización en vivo (mid)"
                else:
                    prima_actual = clean_num(pos.get("UltimoPrecio"))
                    fuente_precio_pos = "cartera IOL (puede estar desactualizado)"

                # Histórico de precio de ESTA opción puntual (bandas de
                # máximo/mínimo/apertura/cierre por rueda), no del subyacente.
                hist_opt = pd.DataFrame()
                hist_opt_raw = None
                if client_pos is not None:
                    try:
                        # +1 día sobre hoy: algunos endpoints de series históricas
                        # tratan el límite superior como exclusivo, lo que haría
                        # que la rueda de hoy nunca aparezca aunque ya esté
                        # disponible. Pedir un día de más es inofensivo (si ese
                        # día no existe todavía, simplemente no viene nada para él).
                        fecha_hasta_str = (datetime.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        fecha_desde_str = (datetime.now() - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
                        hist_opt_raw = client_pos.get_history(ticker, fecha_desde_str, fecha_hasta_str)
                        hist_opt = normalize_option_history(hist_opt_raw)
                    except Exception:
                        hist_opt = pd.DataFrame()

                theo_pos = bs_price(S_pos, strike, T_pos, r, hv_pos, typ) if not np.isnan(strike) else np.nan
                iv_pos = implied_vol(prima_actual, S_pos, strike, T_pos, r, typ) if (not np.isnan(prima_actual) and not np.isnan(strike)) else np.nan
                sig_pos = iv_pos if not np.isnan(iv_pos) else hv_pos
                if not np.isnan(strike):
                    delta_pos, gamma_pos, vega_pos, theta_pos, rho_pos, prob_itm_pos = greeks(S_pos, strike, T_pos, r, sig_pos, typ)
                else:
                    delta_pos = gamma_pos = vega_pos = theta_pos = rho_pos = prob_itm_pos = np.nan

                intrinsic_pos = max(S_pos - strike, 0) if typ == "call" else max(strike - S_pos, 0) if not np.isnan(strike) else np.nan
                extrinsic_pos = (prima_actual - intrinsic_pos) if (not np.isnan(prima_actual) and not np.isnan(intrinsic_pos)) else np.nan

                liquidez_pos = np.nan
                if not (np.isnan(vol_compra_op) and np.isnan(vol_venta_op)):
                    score_liq = 50.0
                    vol_total = np.nansum([vol_compra_op, vol_venta_op])
                    if vol_total >= 500: score_liq += 25
                    elif vol_total >= 100: score_liq += 10
                    elif vol_total <= 5: score_liq -= 20
                    if not np.isnan(compra_op) and not np.isnan(venta_op) and venta_op > 0:
                        spread_pct = (venta_op - compra_op) / venta_op * 100
                        if spread_pct <= 3: score_liq += 25
                        elif spread_pct <= 8: score_liq += 10
                        elif spread_pct >= 20: score_liq -= 25
                    liquidez_pos = float(np.clip(score_liq, 0, 100))

                valor_esperado_mantener, valor_cierre_ahora = position_expected_value(
                    typ, strike, pos.get("Cantidad"), prima_actual, S_pos, T_pos, prob_pos.get("Mu ajustada", 0.0), hv_pos
                )

                accion, score_pos, razones_pos, pnl_pct = recommend_option_action(
                    typ, pos.get("Cantidad"), dias_venc_pos, extrinsic_pos, prima_actual, pos.get("PPC"),
                    prob_pos, veredicto_pos, valor_esperado_mantener, valor_cierre_ahora, liquidez_pos,
                )

                resultados.append({
                    "Ticker": ticker,
                    "Vencimiento": fecha_venc_pos.strftime("%d/%m/%Y") if fecha_venc_pos is not None else f"~{dias_venc_pos}d (manual)",
                    "Subyacente": subyacente,
                    "Tipo": typ.upper(),
                    "Cantidad": pos.get("Cantidad"),
                    "PPC": pos.get("PPC"),
                    "Último": prima_actual,
                    "P&L %": pnl_pct,
                    "Veredicto técnico": veredicto_pos,
                    "Acción sugerida": accion,
                    "Razones": " · ".join(razones_pos),
                })

                detalles_extra[ticker] = dict(
                    subyacente=subyacente, dias_venc_pos=dias_venc_pos, fuente_precio_pos=fuente_precio_pos,
                    prob_pos=prob_pos, rsi_pos=rsi_pos, macd_pos=macd_pos, macd_signal_pos=macd_signal_pos,
                    macd_hist_pos=macd_hist_pos, sma20_pos=sma20_pos, bb_up_pos=bb_up_pos, bb_dn_pos=bb_dn_pos,
                    ema50_pos=ema50_pos, adx_pos=adx_pos, plus_di_pos=plus_di_pos, minus_di_pos=minus_di_pos,
                    close_pos=close_pos, volumen_pos_series=volumen_pos_series,
                    delta_pos=delta_pos, gamma_pos=gamma_pos, vega_pos=vega_pos, theta_pos=theta_pos,
                    intrinsic_pos=intrinsic_pos, extrinsic_pos=extrinsic_pos, theo_pos=theo_pos,
                    compra_op=compra_op, venta_op=venta_op, liquidez_pos=liquidez_pos,
                    valor_esperado_mantener=valor_esperado_mantener, valor_cierre_ahora=valor_cierre_ahora,
                    hist_opt=hist_opt,
                )
                if hist_opt_raw is not None:
                    st.session_state.setdefault("hist_opt_raw_debug", {})[ticker] = hist_opt_raw

            if not resultados:
                st.warning("No pude procesar las posiciones encontradas (faltan datos de precio o strike). Revisá 'Debug IOL'.")
            else:
                res_df = pd.DataFrame(resultados)
                st.dataframe(
                    res_df.drop(columns=["Razones"]).style.format(
                        {"PPC": "{:.2f}", "Último": "{:.2f}", "P&L %": "{:.1f}", "Cantidad": "{:.0f}"}, na_rep="",
                    ),
                    use_container_width=True,
                )

                st.markdown("#### Riesgo de cartera por vencimiento")
                st.caption("Vista consolidada de toda la cartera de opciones -- no posición por posición -- para ver si el riesgo de vencimiento está concentrado en pocos días.")
                riesgo = portfolio_expiry_risk(resultados, detalles_extra)
                if riesgo:
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Valor nocional total", f"{riesgo['nocional_total']:,.0f}")
                    rc2.metric("Valor extrínseco total", f"{riesgo['extrinsico_total']:,.0f}")
                    rc3.metric("Theta total (por día)", f"{riesgo['theta_total']:,.0f}")
                    if not np.isnan(riesgo["dias_min"]):
                        st.caption(f"El vencimiento más próximo de la cartera es en {int(riesgo['dias_min'])} días.")

                    bucket_df = pd.DataFrame([{"Horizonte": k, "Nocional en riesgo": v} for k, v in riesgo["buckets"].items()])
                    st.dataframe(bucket_df.style.format({"Nocional en riesgo": "{:,.0f}"}), use_container_width=True)

                    total_nocional_abs = sum(abs(v) for v in riesgo["buckets"].values())
                    if total_nocional_abs > 0:
                        pct_corto = (riesgo["buckets"]["≤5 días"] + riesgo["buckets"]["6-15 días"]) / total_nocional_abs * 100
                        if pct_corto > 60:
                            st.warning(f"El {pct_corto:.0f}% del valor nocional de la cartera vence en menos de 15 días: riesgo de vencimiento concentrado en el corto plazo.")
                        else:
                            st.caption(f"{pct_corto:.0f}% del valor nocional vence en menos de 15 días.")
                    st.caption("Theta total: cuánto se espera que cambie el valor conjunto de la cartera por día, sólo por el paso del tiempo, sin cambios de precio del subyacente (positivo favorece a quien vendió/lanzó, negativo a quien compró).")

                st.markdown("#### Detalle por posición")
                st.caption("Cada posición se analiza con la misma información que el resto de la app: pronóstico Busa AI, RSI/MACD/Bollinger/ADX del subyacente, griegas de la opción, liquidez puntual y una comparación de valor esperado entre mantener y cerrar ahora.")
                for _, r_row in res_df.iterrows():
                    ticker = r_row["Ticker"]
                    extra = detalles_extra.get(ticker, {})
                    css_class = "score-good" if r_row["Acción sugerida"] == "MANTENER" else "score-bad" if r_row["Acción sugerida"] == "VENDER" else "score-mid"
                    with st.container(border=True):
                        st.markdown(
                            f"##### {ticker} ({r_row['Subyacente']}, {r_row['Tipo']}) — "
                            f"<span class='{css_class}' style='font-size:20px;'>{r_row['Acción sugerida']}</span>",
                            unsafe_allow_html=True,
                        )
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Cantidad", "" if pd.isna(r_row["Cantidad"]) else f"{r_row['Cantidad']:.0f}")
                        c2.metric("PPC", "" if pd.isna(r_row["PPC"]) else f"{r_row['PPC']:.2f}")
                        c3.metric("Precio actual", "" if pd.isna(r_row["Último"]) else f"{r_row['Último']:.2f}")
                        c4.metric("P&L %", "" if pd.isna(r_row["P&L %"]) else f"{r_row['P&L %']:.1f}%")
                        c5.metric("Días venc.", extra.get("dias_venc_pos", ""))
                        st.caption(f"Vencimiento: {r_row['Vencimiento']} | Precio de referencia: {extra.get('fuente_precio_pos', '')}")

                        if extra:
                            st.markdown("**Griegas de la posición (y Black-Scholes)**")
                            g0, g1, g2, g3, g4, g5 = st.columns(6)
                            theo_val = extra.get("theo_pos", np.nan)
                            g0.metric("Black-Scholes", "" if pd.isna(theo_val) else f"{theo_val:,.2f}")
                            g1.metric("Delta", "" if pd.isna(extra.get("delta_pos", np.nan)) else f"{extra['delta_pos']:.3f}")
                            g2.metric("Gamma", "" if pd.isna(extra.get("gamma_pos", np.nan)) else f"{extra['gamma_pos']:.4f}")
                            g3.metric("Theta diario", "" if pd.isna(extra.get("theta_pos", np.nan)) else f"{extra['theta_pos']:.2f}")
                            g4.metric("Vega x1%", "" if pd.isna(extra.get("vega_pos", np.nan)) else f"{extra['vega_pos']:.2f}")
                            g5.metric("Extrínseco", "" if pd.isna(extra.get("extrinsic_pos", np.nan)) else f"{extra['extrinsic_pos']:.2f}")
                            if not pd.isna(theo_val) and theo_val > 0 and not pd.isna(r_row["Último"]):
                                dif_bs = (r_row["Último"] / theo_val - 1) * 100
                                st.caption(f"Precio de mercado vs. Black-Scholes: {dif_bs:+.1f}% ({'cara' if dif_bs > 0 else 'barata'} respecto del valor teórico, según la volatilidad histórica usada).")

                            st.markdown(f"**Contexto técnico de {extra['subyacente']}**")
                            pp = extra["prob_pos"]
                            st.write(f"🎯 Busa AI: Sube {pp['Sube']:.0%} · Baja {pp['Baja']:.0%} · Lateral {pp['Lateral']:.0%}")
                            st.write(f"📈 {rsi_expert_reading(extra['rsi_pos'])}")
                            st.write(f"📊 {macd_expert_reading(extra['macd_pos'], extra['macd_signal_pos'], extra['macd_hist_pos'])}")
                            st.write(f"📉 {bollinger_expert_reading(extra['close_pos'], extra['sma20_pos'], extra['bb_up_pos'], extra['bb_dn_pos'])}")
                            st.write(f"📐 {trend_expert_reading(extra['close_pos'], extra['ema50_pos'])}")
                            st.write(f"🧭 {adx_expert_reading(extra['adx_pos'], extra['plus_di_pos'], extra['minus_di_pos'])}")
                            if extra.get("volumen_pos_series") is not None:
                                st.write(f"📦 {volume_expert_reading(extra['volumen_pos_series'], extra['close_pos'])}")

                            st.markdown("**Valor esperado: mantener vs. cerrar ahora**")
                            v1, v2 = st.columns(2)
                            vem = extra.get("valor_esperado_mantener", np.nan)
                            vca = extra.get("valor_cierre_ahora", np.nan)
                            v1.metric("Cerrar ahora", "" if pd.isna(vca) else f"{vca:,.0f}")
                            v2.metric("Valor esperado (mantener)", "" if pd.isna(vem) else f"{vem:,.0f}")
                            liq_txt = f"Liquidez de esta opción puntual: {extra['liquidez_pos']:.0f}/100" if not pd.isna(extra.get("liquidez_pos", np.nan)) else "Liquidez de esta opción puntual: sin datos en vivo"
                            if not pd.isna(extra.get("compra_op", np.nan)) and not pd.isna(extra.get("venta_op", np.nan)):
                                liq_txt += f" | Compra {extra['compra_op']:.2f} / Venta {extra['venta_op']:.2f}"
                            st.caption(liq_txt)

                            hist_opt_pos = extra.get("hist_opt", pd.DataFrame())
                            if hist_opt_pos is not None and not hist_opt_pos.empty:
                                st.markdown("**Bandas de precio de la opción (histórico propio, no el del subyacente)**")
                                stats_opt = option_history_stats(hist_opt_pos, donchian_window_cartera)
                                if stats_opt:
                                    s1, s2, s3, s4 = st.columns(4)
                                    s1.metric(f"Máximo ({stats_opt['n_ruedas']}r)", f"{stats_opt['maximo']:,.2f}")
                                    s2.metric(f"Mínimo ({stats_opt['n_ruedas']}r)", f"{stats_opt['minimo']:,.2f}")
                                    s3.metric("Desde el máximo", "" if pd.isna(stats_opt["desde_maximo_pct"]) else f"{stats_opt['desde_maximo_pct']:.1f}%")
                                    s4.metric("Posición en el rango", "" if pd.isna(stats_opt["pos_en_rango_pct"]) else f"{stats_opt['pos_en_rango_pct']:.0f}%")
                                    st.caption("'Posición en el rango': 0% = tocando el mínimo del período, 100% = tocando el máximo.")
                                ppc_val = r_row["PPC"] if not pd.isna(r_row["PPC"]) else None
                                st.plotly_chart(build_option_history_figure(hist_opt_pos, ppc_val, ticker, donchian_window_cartera), use_container_width=True, config={"displaylogo": False})
                                st.caption("Línea punteada amarilla: tu PPC. Líneas grises finas: Canal de Donchian (máx./mín. de la ventana elegida arriba) — no es Bollinger porque la prima de una opción decae con el tiempo (theta) y no revierte a una media, a diferencia del subyacente.")
                                if "Volume" in hist_opt_pos.columns and hist_opt_pos["Volume"].notna().any():
                                    st.plotly_chart(build_volume_figure(hist_opt_pos, ticker), use_container_width=True, config={"displaylogo": False})
                            else:
                                st.caption("No pude traer el histórico de precio de esta opción puntual (dato experimental, recién conectado -- revisá 'Debug IOL — Histórico de opciones' más abajo si esperabas verlo).")

                        st.markdown("**Por qué esta recomendación**")
                        st.caption(r_row["Razones"])

                st.caption("Recomendación basada en reglas explícitas + comparación de valor esperado bajo el modelo de pronóstico vigente. No es asesoramiento financiero personalizado — la decisión final es tuya.")

    with st.expander("Debug IOL — Cartera", expanded=False):
        if "portfolio_raw" in st.session_state:
            st.json(st.session_state["portfolio_raw"])
        else:
            st.info("Sin respuesta cruda de cartera todavía.")

    with st.expander("Debug IOL — Histórico de opciones", expanded=False):
        hist_debug = st.session_state.get("hist_opt_raw_debug", {})
        if hist_debug:
            for tk, raw_h in hist_debug.items():
                st.write(f"**{tk}**")
                st.json(raw_h)
        else:
            st.info("Sin respuesta cruda de histórico de opciones todavía. Se completa al traer la cartera.")

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
