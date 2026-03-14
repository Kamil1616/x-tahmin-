import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Android 11; Mobile; rv:120.0) Gecko/120.0 Firefox/120.0",
    "Referer": "https://www.sofascore.com",
    "Accept": "application/json",
}

BASE_URL = "https://api.sofascore.com/api/v1"

@st.cache_data(ttl=600)
def search_teams(query):
    if len(query) < 3:
        return []
    try:
        resp = requests.get(f"{BASE_URL}/search/teams?q={query}", headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            return resp.json().get("results", [])[:6]
        return []
    except:
        return []

@st.cache_data(ttl=900)
def get_last_matches(team_id):
    try:
        resp = requests.get(f"{BASE_URL}/team/{team_id}/events/last/0", headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return []
        events = resp.json().get("events", [])
        matches = []
        for e in events:
            if e.get("status", {}).get("type") != "finished":
                continue
            home = e["homeTeam"]
            away = e["awayTeam"]
            is_home = home["id"] == team_id
            sh = e["homeScore"].get("normaltime", 0)
            sa = e["awayScore"].get("normaltime", 0)
            matches.append({
                "is_home": is_home,
                "goals_scored": sh if is_home else sa,
                "goals_conceded": sa if is_home else sh,
                "result": 1 if (sh > sa and is_home) or (sa > sh and not is_home) else 0 if sh == sa else -1
            })
        return matches  # timestamp yoktu, sıralamayı kaldırdık (son maçlar zaten son sıralı gelir)
    except:
        return []

def analyze(matches, home_only=False, away_only=False, n=6):
    filtered = matches
    if home_only:
        filtered = [m for m in matches if m["is_home"]]
    elif away_only:
        filtered = [m for m in matches if not m["is_home"]]
    recent = filtered[:n]
    if len(recent) < 3:
        return None
    gs = np.mean([m["goals_scored"] for m in recent])
    gc = np.mean([m["goals_conceded"] for m in recent])
    return {"scored": gs, "conceded": gc}

def predict(h_stats, a_stats):
    if not h_stats or not a_stats:
        return None
    lh = (h_stats["scored"] * 1.1 + a_stats["conceded"] * 0.9) / 2
    la = (a_stats["scored"] * 0.9 + h_stats["conceded"] * 1.1) / 2
    probs = {"1":0, "X":0, "2":0, "over_0.5":0, "over_1.5":0, "over_2.5":0, "over_3.5":0}
    for i in range(8):
        for j in range(8):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la)
            if i > j: probs["1"] += p
            elif i == j: probs["X"] += p
            else: probs["2"] += p
            tg = i + j
            if tg > 0.5: probs["over_0.5"] += p
            if tg > 1.5: probs["over_1.5"] += p
            if tg > 2.5: probs["over_2.5"] += p
            if tg > 3.5: probs["over_3.5"] += p
    odds = {k: round(1 / v * 1.08, 2) if v > 0.02 else "—" for k, v in probs.items()}
    return {
        "probs": {k: f"{v*100:.1f}%" for k, v in probs.items()},
        "odds": odds,
        "xg": f"{lh:.2f} – {la:.2f}"
    }

st.title("Maç Tahmin Aracı – Son 6 Maç")

col1, col2 = st.columns(2)

with col1:
    home_q = st.text_input("Ev sahibi takım ara", key="home_input")
    home_res = search_teams(home_q)
    home_options = [t.get("name", "Bilinmeyen Takım") for t in home_res]
    home_sel = st.selectbox("Ev sahibi seç", options=home_options if home_options else ["Takım ara..."], key="home_sel")

with col2:
    away_q = st.text_input("Deplasman takım ara", key="away_input")
    away_res = search_teams(away_q)
    away_options = [t.get("name", "Bilinmeyen Takım") for t in away_res]
    away_sel = st.selectbox("Deplasman seç", options=away_options if away_options else ["Takım ara..."], key="away_sel")

if st.button("Tahmin Yap"):
    if home_sel in ["Takım ara...", "Bilinmeyen Takım"] or away_sel in ["Takım ara...", "Bilinmeyen Takım"]:
        st.warning("Lütfen iki geçerli takımı seçin.")
    else:
        # Seçilen isme göre takım objesini bul
        home_t = next((t for t in home_res if t.get("name") == home_sel), None)
        away_t = next((t for t in away_res if t.get("name") == away_sel), None)
        
        if home_t is None or away_t is None:
            st.error("Seçilen takımın bilgileri alınamadı. Tekrar aratmayı deneyin.")
        else:
            with st.spinner("Maç verileri çekiliyor..."):
                h_matches = get_last_matches(home_t["id"])
                a_matches = get_last_matches(away_t["id"])
                
                h_home_stats = analyze(h_matches, home_only=True) or analyze(h_matches)
                a_away_stats = analyze(a_matches, away_only=True) or analyze(a_matches)
                
                if h_home_stats is None or a_away_stats is None:
                    st.warning("Yeterli maç verisi yok.")
                else:
                    pred = predict(h_home_stats, a_away_stats)
                    if pred:
                        st.success("Tahmin hazır!")
                        st.write(f"**Beklenen gol (xG)**: {pred['xg']}")
                        
                        st.subheader("Maç Sonucu")
                        for res in ["1", "X", "2"]:
                            st.write(f"**{res}**: {pred['probs'][res]} → oran ≈ {pred['odds'][res]}")
                        
                        st.subheader("Üst / Alt")
                        for ov in ["over_0.5", "over_1.5", "over_2.5", "over_3.5"]:
                            label = ov.replace("over_", "Üst ").replace("_", ".")
                            st.write(f"**{label}**: {pred['probs'][ov]} → oran ≈ {pred['odds'][ov]}")
                    else:
                        st.error("Tahmin hesaplanamadı.")
