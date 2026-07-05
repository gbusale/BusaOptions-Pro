
import requests
import tomllib
from pathlib import Path

class IOLAuthError(Exception):
    pass

class IOLApiError(Exception):
    pass

class IOLClient:
    BASE_URL = "https://api.invertironline.com"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.access_token = None

    @classmethod
    def from_config(cls, path: str = "config/credentials.toml"):
        """
        En Streamlit Cloud lee st.secrets.
        En local lee config/credentials.toml.
        """
        try:
            import streamlit as st
            if "IOL" in st.secrets:
                return cls(st.secrets["IOL"]["username"], st.secrets["IOL"]["password"])
        except Exception:
            pass

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("No encontré credenciales locales ni Streamlit Secrets.")
        with p.open("rb") as f:
            cfg = tomllib.load(f)
        return cls(cfg["IOL"]["username"], cfg["IOL"]["password"])

    def login(self):
        resp = requests.post(
            f"{self.BASE_URL}/token",
            data={
                "username": self.username,
                "password": self.password,
                "grant_type": "password",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        if resp.status_code != 200:
            raise IOLAuthError(f"HTTP {resp.status_code}: {resp.text[:600]}")

        payload = resp.json()
        self.access_token = payload.get("access_token")
        if not self.access_token:
            raise IOLAuthError(f"No vino access_token. Respuesta: {payload}")
        return payload

    def get(self, endpoint: str):
        if not self.access_token:
            self.login()

        resp = requests.get(
            f"{self.BASE_URL}{endpoint}",
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30,
        )

        if resp.status_code == 401:
            self.login()
            resp = requests.get(
                f"{self.BASE_URL}{endpoint}",
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=30,
            )

        if not (200 <= resp.status_code < 300):
            raise IOLApiError(f"GET {endpoint} -> HTTP {resp.status_code}: {resp.text[:800]}")

        return resp.json()

    def get_options(self, simbolo: str, mercado: str = "bCBA"):
        return self.get(f"/api/v2/{mercado}/Titulos/{simbolo}/Opciones")
