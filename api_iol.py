import requests, tomllib
from pathlib import Path
class IOLAuthError(Exception): pass
class IOLApiError(Exception): pass
class IOLClient:
    BASE_URL='https://api.invertironline.com'
    def __init__(self,u,p): self.username=u; self.password=p; self.access_token=None
    @classmethod
    def from_config(cls,path='config/credentials.toml'):
        try:
            import streamlit as st
            if 'IOL' in st.secrets: return cls(st.secrets['IOL']['username'], st.secrets['IOL']['password'])
        except Exception: pass
        p=Path(path)
        if not p.exists(): raise FileNotFoundError('No encontré credenciales locales ni Streamlit Secrets.')
        with p.open('rb') as f: cfg=tomllib.load(f)
        return cls(cfg['IOL']['username'], cfg['IOL']['password'])
    def login(self):
        r=requests.post(self.BASE_URL+'/token',data={'username':self.username,'password':self.password,'grant_type':'password'},headers={'Content-Type':'application/x-www-form-urlencoded'},timeout=20)
        if r.status_code!=200: raise IOLAuthError(f'HTTP {r.status_code}: {r.text[:600]}')
        js=r.json(); self.access_token=js.get('access_token')
        if not self.access_token: raise IOLAuthError(f'No vino access_token: {js}')
        return js
    def get(self,endpoint):
        if not self.access_token: self.login()
        r=requests.get(self.BASE_URL+endpoint,headers={'Authorization':f'Bearer {self.access_token}'},timeout=30)
        if r.status_code==401:
            self.login(); r=requests.get(self.BASE_URL+endpoint,headers={'Authorization':f'Bearer {self.access_token}'},timeout=30)
        if not 200<=r.status_code<300: raise IOLApiError(f'HTTP {r.status_code}: {r.text[:800]}')
        return r.json()
    def get_options(self,simbolo,mercado='bCBA'):
        return self.get(f'/api/v2/{mercado}/Titulos/{simbolo}/Opciones')
