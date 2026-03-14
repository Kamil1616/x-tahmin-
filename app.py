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
    if len(query) < 3: return []
    try:
        resp = requests.get(f"{BASE_URL}/search/teams?q={query}", headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            return resp.json().get("results", [])[:6]
    except: return []
    return []

@st.cache_data(ttl=900)
def get_last_matches(team_id):
    try:
        resp = requests.get(f"{BASE_URL}/team/{team_id}/events/last/0", headers=HEADERS, timeout=12)
        if resp.status_code != 200: return []
        events = resp.json().get("events", [])
        matches = []
        for e in events:
            if e.get("status", {}).get("type") != "finished": continue
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
        matches.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return matches
    except: return []

def analyze(matches, home_only=False, away_only=False, n=6):
    filtered = [m for m in matches if (home_only and m["is_home"]) or (away_only and not m["is_home"]) or (not home_only and not away_only)]
    recent = filtered[:n]
    if len(recent) < 3: return None
    gs = np.mean([m["goals_scored"] for m in recent])
    gc = np.mean([m["goals_conceded"] for m in recent])
    return {"scored": gs, "conceded": gc}

def predict(h_stats, a_stats):
    if not h_stats or not a_stats: return None
    lh = (h_stats["scored"] * 1.1 + a_stats["conceded"] * 0.9) / 2
    la = (a_stats["scored"] * 0.9 + h_stats["conceded"] * 1.1) / 2
    probs = {"1":0, "X":0, "2":0, "over_0.5":0, "over_1.5":0, "over_2.5":0, "over_3.5":0}
    for i in range(8):
        for j in range(8):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la)
            if i > j: probs["1"] += p
            elif i == j: probs["X"] += p
            else: probs["2"] += p
            tg = i+j
            if tg > 0.5: probs["over_0.5"] += p
            if tg > 1.5: probs["over_1.5"] += p
            if tg > 2.5: probs["over_2.5"] += p
            if tg > 3.5: probs["over_3.5"] += p
    odds = {k: round(1/v * 1.08, 2) if v>0.02 else "—" for k,v in probs.items()}
    return {"probs": {k:f"{v*100:.1f}%" for k,v in probs.items()}, "odds": odds, "xg": f"{lh:.2f}–{la:.2f}"}

st.title("Maç Tahmin Aracı – Son 6 Maç")

col1, col2 = st.columns(2)
with col1:
    home_q = st.text_input("Ev sahibi ara", key="h")
    home_res = search_teams(home_q)
    home_sel = st.selectbox("Ev sahibi", [t["name"] for t in home_res] if home_res else [], key="hs")

with col2:
    away_q = st.text_input("Deplasman ara", key="a")
    away_res = search_teams(away_q)
    away_sel = st.selectbox("Deplasman", [t["name"] for t in away_res] if away_res else [], key="as")

if st.button("Tahmin Yap"):
    if not home_sel or not away_sel:
        st.warning("Takımları seç")
    else:
        home_t = next(t for t in home_res if t["name"] == home_sel)
        away_t = next(t for t in away_res if t["name"] == away_sel)
        h_matches = get_last_matches(home_t["id"])
        a_matches = get_last_matches(away_t["id"])
        h_home = analyze(h_matches, home_only=True)
        a_away = analyze(a_matches, away_only=True)
        if not h_home: h_home = analyze(h_matches)
        if not a_away: a_away = analyze(a_matches)
        pred = predict(h_home, a_away)
        if pred:
            st.write(f"xG: {pred['xg']}")
            st.subheader("Maç Sonucu")
            for r in ["1","X","2"]: st.write(f"{r}: {pred['probs'][r]} → {pred['odds'][r]}")
            st.subheader("Üst/Alt")
            for ov in ["over_0.5","over_1.5","over_2.5","over_3.5"]:
                label = ov.replace("over_","Üst ").replace("_",".")
                st.write(f"{label}: {pred['probs'][ov]} → {pred['odds'][ov]}")
        else:
            st.error("Veri yetersiz")
