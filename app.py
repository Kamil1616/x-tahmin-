import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Android 11; Mobile; rv:120.0) Gecko/120.0 Firefox/120.0",
    "Referer": "https://www.sofascore.com",
    "Accept": "application/json",
}

BASE_URL = "https://api.sofascore.com/api/v1"

# ================== FİKSTÜR ÇEKME ==================
@st.cache_data(ttl=1800)  # 30 dk cache
def get_fixtures(date_str):
    url = f"{BASE_URL}/sport/football/scheduled-events/{date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()  # direkt array dönüyor
        return []
    except:
        return []

# ================== SON MAÇLAR & TAHMİN (eski fonksiyonlar) ==================
@st.cache_data(ttl=900)
def get_last_matches(team_id):
    try:
        resp = requests.get(f"{BASE_URL}/team/{team_id}/events/last/0", headers=HEADERS, timeout=15)
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
                "goals_conceded": sa if is_home else sh
            })
        return matches
    except:
        return []

def analyze(matches, home_only=False, away_only=False, n=6):
    filtered = [m for m in matches if (home_only and m["is_home"]) or (away_only and not m["is_home"]) or (not home_only and not away_only)]
    recent = filtered[:n]
    if len(recent) < 3: return None
    gs = np.mean([m["goals_scored"] for m in recent])
    gc = np.mean([m["goals_conceded"] for m in recent])
    return {"scored": gs, "conceded": gc}

def predict_match(home_id, away_id):
    h_matches = get_last_matches(home_id)
    a_matches = get_last_matches(away_id)
    h_stats = analyze(h_matches, home_only=True) or analyze(h_matches)
    a_stats = analyze(a_matches, away_only=True) or analyze(a_matches)
    if not h_stats or not a_stats: return None

    lh = (h_stats["scored"] * 1.1 + a_stats["conceded"] * 0.9) / 2
    la = (a_stats["scored"] * 0.9 + h_stats["conceded"] * 1.1) / 2
    probs = {"1":0, "X":0, "2":0, "over_2.5":0}
    for i in range(9):
        for j in range(9):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la)
            if i > j: probs["1"] += p
            elif i == j: probs["X"] += p
            else: probs["2"] += p
            if i+j > 2.5: probs["over_2.5"] += p
    odds = {k: round(1/v*1.08, 2) if v>0.02 else "—" for k,v in probs.items()}
    return {
        "probs": {k: f"{v*100:.1f}%" for k,v in probs.items()},
        "odds": odds,
        "xg": f"{lh:.2f} – {la:.2f}"
    }

# ================== STREAMLIT ARAYÜZ ==================
st.title("📅 Fikstür Tahmin Aracı – Son 6 Maç")

# Tarih seçimi
today = datetime.now().date()
dates = {
    "Bugün": today.strftime("%Y-%m-%d"),
    "Yarın": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
    "Yarın+1": (today + timedelta(days=2)).strftime("%Y-%m-%d")
}
selected_date_label = st.sidebar.selectbox("Tarih Seç", list(dates.keys()))
selected_date = dates[selected_date_label]

fixtures = get_fixtures(selected_date)

if not fixtures:
    st.error("Bugün için maç bulunamadı veya API geçici olarak yavaş.")
    st.stop()

st.subheader(f"{selected_date_label} ({selected_date}) – {len(fixtures)} maç")

# En yüksek olasılıklı maçlar butonu
if st.button("🔥 En Yüksek Olasılıklı Maçları Göster (Üst 2.5)", type="primary"):
    with st.spinner("Tüm maçlar hesaplanıyor... (biraz sürebilir)"):
        results = []
        for event in fixtures:
            if event.get("status", {}).get("type") == "finished": continue
            home = event.get("homeTeam", {})
            away = event.get("awayTeam", {})
            if not home or not away: continue
            pred = predict_match(home.get("id"), away.get("id"))
            if pred:
                over_pct = float(pred["probs"]["over_2.5"].replace("%",""))
                results.append({
                    "lig": event.get("tournament", {}).get("name", "Lig"),
                    "ev": home.get("name", "???"),
                    "dep": away.get("name", "???"),
                    "saat": datetime.fromtimestamp(event["startTimestamp"]).strftime("%H:%M"),
                    "over_2.5": pred["probs"]["over_2.5"],
                    "pct": over_pct
                })
        if results:
            results.sort(key=lambda x: x["pct"], reverse=True)
            for r in results[:12]:
                st.write(f"**{r['saat']}** | {r['lig']} | {r['ev']} - {r['dep']} → **Üst 2.5: {r['over_2.5']}**")
        else:
            st.info("Henüz yeterli veri yok.")

# Normal fikstür listesi
for event in fixtures:
    if event.get("status", {}).get("type") == "finished": continue
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    saat = datetime.fromtimestamp(event["startTimestamp"]).strftime("%H:%M")
    lig = event.get("tournament", {}).get("name", "—")

    with st.expander(f"⏰ {saat} | {lig} | {home.get('name','?')} - {away.get('name','?')}"):
        if st.button("Tahmin Hesapla", key=f"btn_{event.get('id')}"):
            with st.spinner("Son 6 maç çekiliyor..."):
                pred = predict_match(home.get("id"), away.get("id"))
                if pred:
                    st.write(f"**xG**: {pred['xg']}")
                    st.write("**Maç Sonucu**")
                    for r in ["1","X","2"]:
                        st.write(f"{r}: {pred['probs'][r]} → oran ≈ {pred['odds'][r]}")
                    st.write("**Üst 2.5**")
                    st.write(f"{pred['probs']['over_2.5']} → oran ≈ {pred['odds']['over_2.5']}")
                else:
                    st.warning("Yeterli maç verisi yok.")

st.caption("Not: Tahminler sadece son 6 iç saha/deplasman maçlarına göre hesaplanır. Gerçek bahis için sakatlık, hava vs. de önemli.")
