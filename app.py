import json
from pathlib import Path
from datetime import datetime
import streamlit as st, pandas as pd, numpy as np, yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from api_iol import IOLClient, IOLAuthError, IOLApiError

st.set_page_config(page_title='BusaOptions Mobile', layout='wide')
st.markdown('''<style>.block-container{padding-top:.8rem;padding-left:.55rem;padding-right:.55rem}div[data-testid="stMetricValue"]{font-size:22px}.prob-row{margin:18px 0 22px}.prob-head{display:flex;justify-content:space-between;margin-bottom:6px}.prob-name,.prob-val{font-weight:800;font-size:20px;color:#f9fafb}.prob-bg{height:28px;background:#1f2937;border-radius:16px;overflow:hidden}.prob-fill{height:28px;border-radius:16px}@media(max-width:768px){h1{font-size:1.45rem!important}h2,h3{font-size:1.08rem!important}button{width:100%}.prob-name,.prob-val{font-size:17px}.prob-bg,.prob-fill{height:24px}}</style>''',unsafe_allow_html=True)
st.title('BusaOptions Mobile')
st.caption('IOL por botón + Black-Scholes + VI/VH + griegas + Score Busa.')
TICKERS={'GGAL':{'local':'GGAL.BA','iol':'GGAL'},'YPF':{'local':'YPFD.BA','iol':'YPFD'}}
USAGE_FILE=Path('data/iol_api_usage.json'); LIMIT=25000

def month(): return datetime.now().strftime('%Y-%m')
def load_usage():
    if not USAGE_FILE.exists(): return {'month':month(),'calls':0,'updates':0,'last_update':None}
    try: d=json.loads(USAGE_FILE.read_text(encoding='utf-8'))
    except Exception: d={'month':month(),'calls':0,'updates':0,'last_update':None}
    return d if d.get('month')==month() else {'month':month(),'calls':0,'updates':0,'last_update':None}
def save_usage(d): USAGE_FILE.parent.mkdir(exist_ok=True); USAGE_FILE.write_text(json.dumps(d,indent=2),encoding='utf-8')
def add_calls(n):
    d=load_usage(); d['calls']=int(d.get('calls',0))+n; d['updates']=int(d.get('updates',0))+1; d['last_update']=datetime.now().strftime('%d/%m/%Y %H:%M:%S'); save_usage(d)
def reset_usage(): save_usage({'month':month(),'calls':0,'updates':0,'last_update':None})
@st.cache_data(ttl=180)
def hist(ticker, period):
    df=yf.download(ticker,period=period,progress=False,auto_adjust=True)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    return df.dropna()
def num(x):
    if x is None or pd.isna(x): return np.nan
    if isinstance(x,str):
        x=x.strip();
        if not x: return np.nan
        if ',' in x: x=x.replace('.','').replace(',','.')
    try: return float(x)
    except Exception: return np.nan
def bs(S,K,T,r,sig,typ):
    if min(S,K,T,sig)<=0: return np.nan
    d1=(np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if typ=='call' else K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
def greeks(S,K,T,r,sig,typ):
    if min(S,K,T,sig)<=0: return [np.nan]*6
    d1=(np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T); gamma=norm.pdf(d1)/(S*sig*np.sqrt(T)); vega=S*norm.pdf(d1)*np.sqrt(T)/100
    if typ=='call': return norm.cdf(d1),gamma,vega,(-S*norm.pdf(d1)*sig/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365,K*T*np.exp(-r*T)*norm.cdf(d2)/100,norm.cdf(d2)
    return -norm.cdf(-d1),gamma,vega,(-S*norm.pdf(d1)*sig/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365,-K*T*np.exp(-r*T)*norm.cdf(-d2)/100,norm.cdf(-d2)
def ivol(price,S,K,T,r,typ):
    try: return brentq(lambda s: bs(S,K,T,r,s,typ)-price,.001,5)
    except Exception: return np.nan
def prob(close,horizon,lateral,lookback):
    rets=close.pct_change().dropna(); lb=rets.tail(int(lookback)); S=float(close.iloc[-1]); hv=float(lb.std()*np.sqrt(252)); mu=float(lb.mean()*252); T=horizon/252; up=S*(1+lateral); down=S*(1-lateral); m=np.log(S)+(mu-.5*hv**2)*T; sd=hv*np.sqrt(T); pdn=norm.cdf((np.log(down)-m)/sd); pup=1-norm.cdf((np.log(up)-m)/sd)
    return {'VH':hv,'Sube':pup,'Baja':pdn,'Lateral':max(0,1-pup-pdn),'Nivel suba':up,'Nivel baja':down}
def typ(sym):
    s=str(sym).upper(); return 'put' if ('GFGV' in s or 'YPFV' in s or s.endswith('V')) else 'call'
def normalize(raw):
    if isinstance(raw,dict):
        for k in ['opciones','titulos','data','result','items']:
            if isinstance(raw.get(k),list): raw=raw[k]; break
    if not isinstance(raw,list): return pd.DataFrame()
    rows=[]
    for it in raw:
        if not isinstance(it,dict): continue
        titulo=it.get('titulo') if isinstance(it.get('titulo'),dict) else it; cot=it.get('cotizacion') if isinstance(it.get('cotizacion'),dict) else it; puntas=cot.get('puntas') if isinstance(cot.get('puntas'),dict) else {}; simb=titulo.get('simbolo') or it.get('simbolo') or it.get('ticker') or it.get('descripcion') or ''; strike=it.get('precioEjercicio') or it.get('strike') or titulo.get('precioEjercicio') or titulo.get('strike')
        if strike is None or pd.isna(strike):
            import re; m=re.search(r'(\d{3,6})',str(simb)); strike=m.group(1) if m else np.nan
        rows.append({'Ticker':str(simb).upper(),'Tipo':typ(simb),'Strike':num(strike),'Compra':num(puntas.get('precioCompra') or cot.get('precioCompra') or cot.get('compra') or it.get('compra')),'Venta':num(puntas.get('precioVenta') or cot.get('precioVenta') or cot.get('venta') or it.get('venta')),'Último':num(cot.get('ultimoPrecio') or cot.get('ultimo') or cot.get('precio') or it.get('ultimo')),'Volumen':num(cot.get('volumen') or it.get('volumen'))})
    df=pd.DataFrame(rows); return df.dropna(subset=['Strike']).sort_values(['Tipo','Strike','Ticker']) if not df.empty else df
def analyze(df,S,T,r,hv,pup,pdn,mode):
    rows=[]
    for _,rw in df.iterrows():
        t=str(rw.get('Tipo','call')).lower(); K=num(rw.get('Strike')); compra=num(rw.get('Compra')); venta=num(rw.get('Venta')); ult=num(rw.get('Último'))
        prima=(np.nanmean([compra,venta]) if not (np.isnan(compra) and np.isnan(venta)) else ult) if mode=='Promedio compra/venta' else (venta if mode=='Venta' and not np.isnan(venta) else compra if mode=='Compra' and not np.isnan(compra) else ult if not np.isnan(ult) else np.nanmean([compra,venta]))
        theo=bs(S,K,T,r,hv,t); vi=ivol(prima,S,K,T,r,t) if not np.isnan(prima) else np.nan; sig=vi if not np.isnan(vi) else hv; delta,gamma,vega,theta,rho,pitm=greeks(S,K,T,r,sig,t); intr=max(S-K,0) if t=='call' else max(K-S,0); extr=prima-intr if not np.isnan(prima) else np.nan; diff=((prima/theo)-1)*100 if not np.isnan(prima) and theo and theo>0 else np.nan; direc=pup if t=='call' else pdn; score=np.nan; estado='SIN PRECIO'
        if not np.isnan(prima):
            score=50+(25 if not np.isnan(vi) and vi<hv-.10 else 12 if not np.isnan(vi) and vi<hv-.03 else -25 if not np.isnan(vi) and vi>hv+.15 else -12 if not np.isnan(vi) and vi>hv+.07 else 0); score+=10 if 0.25<=abs(delta)<=0.60 else 0; score+=12 if direc>.50 else -8 if direc<.35 else 0; score=max(0,min(100,score)); estado='MUY BARATA vs VH' if not np.isnan(vi) and vi<hv-.10 else 'BARATA vs VH' if not np.isnan(vi) and vi<hv-.03 else 'CARA vs VH' if not np.isnan(vi) and vi>hv+.07 else 'BARATA vs BS' if not np.isnan(diff) and diff<-8 else 'CARA vs BS' if not np.isnan(diff) and diff>8 else 'PRECIO JUSTO'
        rows.append({'Ticker':rw.get('Ticker'),'Tipo':t.upper(),'Strike':K,'Compra':compra,'Venta':venta,'Último':ult,'Prima usada':prima,'Black-Scholes':theo,'Dif % vs BS':diff,'VI %':vi*100 if not np.isnan(vi) else np.nan,'VH %':hv*100,'Spread VI-VH':(vi-hv)*100 if not np.isnan(vi) else np.nan,'Intrínseco':intr,'Extrínseco':extr,'Delta':delta,'Gamma':gamma,'Vega x 1%':vega,'Theta diario':theta,'Prob. ITM %':pitm*100,'Prob. dirección %':direc*100,'Volumen':rw.get('Volumen'),'Score Busa':score,'Estado':estado})
    return pd.DataFrame(rows)
def fmt(df):
    f={c:'{:.2f}' for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    for c in ['Strike','Score Busa','Volumen']:
        if c in f: f[c]='{:.0f}'
    if 'Delta' in f: f['Delta']='{:.3f}'
    if 'Gamma' in f: f['Gamma']='{:.5f}'
    return df.style.format(f,na_rep='')
def prob_bar(label,pct,color):
    pct100=max(0,min(100,pct*100)); st.markdown(f'<div class="prob-row"><div class="prob-head"><span class="prob-name">{label}</span><span class="prob-val">{pct100:.1f}%</span></div><div class="prob-bg"><div class="prob-fill" style="width:{pct100}%;background:{color};"></div></div></div>',unsafe_allow_html=True)

with st.sidebar:
    u=load_usage(); st.header('Actualizar'); st.metric('Consultas mes',f"{u.get('calls',0):,} / {LIMIT:,}"); st.progress(min(1,u.get('calls',0)/LIMIT));
    if u.get('last_update'): st.caption(f"Última API: {u['last_update']}")
    activo=st.selectbox('Activo',['GGAL','YPF']); mode=st.selectbox('Prima usada',['Promedio compra/venta','Venta','Compra','Último'])
    with st.expander('Parámetros',expanded=False):
        period=st.selectbox('Histórico',['6mo','1y','2y','5y'],index=2); lookback=st.number_input('VH ruedas',20,252,60,5); horizon=st.number_input('Horizonte',1,120,20); lateral=st.number_input('Lateral +/- %',0.5,30.0,5.0,.5)/100; r=st.number_input('Tasa caución %',0.0,200.0,20.2,.1)/100; days=st.number_input('Días vencimiento',1,365,52)
    if st.button('🔄 Actualizar mercado (IOL)'):
        try:
            raw=IOLClient.from_config().get_options(TICKERS[activo]['iol']); st.session_state['raw_iol']=raw; st.session_state['options_df']=normalize(raw); st.session_state['last_update']=pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S'); add_calls(2); st.success('Mercado actualizado. Consumo estimado: 2 consultas.')
        except (FileNotFoundError,IOLAuthError,IOLApiError) as e: st.error(str(e))
        except Exception as e: st.error(f'Error: {e}')
    if st.button('🧮 Recalcular análisis'): st.rerun()
    if st.button('Reiniciar contador'): reset_usage(); st.rerun()

h=hist(TICKERS[activo]['local'],period); close=h['Close'].dropna(); S=float(close.iloc[-1]); pr=prob(close,int(horizon),lateral,int(lookback)); hv=pr['VH']; T=days/365
tabs=st.tabs(['Dashboard','Opciones','Probabilidades','Velas'])
df=st.session_state.get('options_df',pd.DataFrame()); analyzed=pd.DataFrame()
if not df.empty: analyzed=analyze(df,S,T,r,hv,pr['Sube'],pr['Baja'],mode)
with tabs[0]:
    st.subheader(f'Dashboard {activo}'); c1,c2=st.columns(2); c1.metric('Precio',f'{S:,.2f}'); c2.metric('VH',f'{hv*100:.1f}%'); c3,c4=st.columns(2); c3.metric('Prob. suba',f"{pr['Sube']:.1%}"); c4.metric('Opciones',len(analyzed) if not analyzed.empty else 0)
    if not analyzed.empty: st.write('### Top oportunidades'); st.dataframe(fmt(analyzed.dropna(subset=['Score Busa']).sort_values('Score Busa',ascending=False).head(3)),use_container_width=True)
    else: st.warning('Tocá Actualizar mercado para cargar opciones.')
with tabs[1]:
    st.subheader(f'Opciones {activo}');
    if 'last_update' in st.session_state: st.caption(f"Última actualización: {st.session_state['last_update']}")
    if analyzed.empty: st.warning('Sin cadena cargada. Tocá Actualizar mercado.')
    else:
        st.write('### Ranking'); st.dataframe(fmt(analyzed.dropna(subset=['Score Busa']).sort_values('Score Busa',ascending=False).head(20)),use_container_width=True); st.write('### CALLS'); st.dataframe(fmt(analyzed[analyzed['Tipo']=='CALL'].sort_values('Strike')),use_container_width=True); st.write('### PUTS'); st.dataframe(fmt(analyzed[analyzed['Tipo']=='PUT'].sort_values('Strike')),use_container_width=True)
with tabs[2]:
    st.subheader('Probabilidades'); prob_bar('Sube',pr['Sube'],'#15803d'); prob_bar('Baja',pr['Baja'],'#b91c1c'); prob_bar('Lateral',pr['Lateral'],'#ca8a04'); c1,c2=st.columns(2); c1.metric('Nivel suba',f"{pr['Nivel suba']:,.2f}"); c2.metric('Nivel baja',f"{pr['Nivel baja']:,.2f}"); st.metric('Precio base',f'{S:,.2f}')
with tabs[3]:
    st.subheader('Velas'); ema20=close.ewm(span=20,adjust=False).mean(); ema50=close.ewm(span=50,adjust=False).mean(); ema200=close.ewm(span=200,adjust=False).mean(); fig=go.Figure(); fig.add_trace(go.Candlestick(x=h.index,open=h['Open'],high=h['High'],low=h['Low'],close=h['Close'],name='Velas')); fig.add_trace(go.Scatter(x=h.index,y=ema20,name='EMA20',line=dict(width=1.2))); fig.add_trace(go.Scatter(x=h.index,y=ema50,name='EMA50',line=dict(width=1.2))); fig.add_trace(go.Scatter(x=h.index,y=ema200,name='EMA200',line=dict(width=1.4))); fig.update_layout(height=520,xaxis_rangeslider_visible=False,template='plotly_dark',hovermode='x unified',margin=dict(l=10,r=10,t=30,b=20)); st.plotly_chart(fig,use_container_width=True)
with st.expander('Debug IOL',expanded=False):
    if 'raw_iol' in st.session_state: st.json(st.session_state['raw_iol'])
    else: st.info('Sin respuesta cruda.')
