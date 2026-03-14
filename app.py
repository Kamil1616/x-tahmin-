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

@st.cache_data(ttl=1800)  # 30 dk cache
def get_fixtures(date_str):
    url = f"{BASE_URL}/sport/football/scheduled-events/{date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return data
            else:
                st.warning(f"API beklenmedik format döndürdü: {type(data)}")
                return []
        else:
            st.warning(f"API hata kodu: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Fikstür çekme hatası: {e}")
        return []

@st.cache_data(ttl=900)
def get_last_matches(team_id):
    try:
        resp = requests.get(f"{BASE_URL}/team/{team_id}/events/last/0", headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        events = resp.json().get("events", [])
        matches = []
        for e in events:
            if not isinstance(e, dict) or e.get("status", {}).get("type") != "finished":
                continue
            home = e.get("homeTeam", {})
            away = e.get("awayTeam", {})
            is_home = home.get("id") == team_id
            sh = e.get("homeScore", {}).get("normaltime", 0)
            sa = e.get("awayScore", {}).get("normaltime", 0)
            matches.append({
                "is_home": is_home,
                "goals_scored": sh if is_home else sa,
                "goals_conceded": sa if is_home else sh
            })
        return matches
    except:
        return []

def analyze(matches, home_only=False, away_only=False, n=6):
    if not matches:
        return None
    filtered = [m for m in matches if (home_only and m["is_home"]) or (away_only and not m["is_home"]) or (not home_only and not away_only)]
    recent = filtered[:n]
    if len(recent) < 3:
        return None
    gs = np.mean([m["goals_scored"] for m in recent])
    gc = np.mean([m["goals_conceded"] for m in recent])
    return {"scored": gs, "conceded": gc}

def predict_match(home_id, away_id):
    h_matches = get_last_matches(home_id)
    a_matches = get_last_matches(away_id)
    h_stats = analyze(h_matches, home_only=True) or analyze(h_matches)
    a_stats = analyze(a_matches, away_only=True) or analyze(a_matches)
    if not h_stats or not a_stats:
        return None
    lh = (h_stats["scored"] * 1.1 + a_stats["conceded"] * 0.9) / 2
    la = (a_stats["scored"] * 0.9 + h_stats["conceded"] * 1.1) / 2
    probs = {"1":0, "X":0, "2":0, "over_2.5":0}
    for i in range(9):
        for j in range(9):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la)
            if i > j: probs["1"] += p
            elif i == j: probs["X"] += p
            else: probs["2"] += p
            if i + j > 2.5: probs["over_2.5"] += p
    odds = {k: round(1 / v * 1.08, 2) if v > 0.02 else "—" for k, v in probs.items()}
    return {
        "probs": {k: f"{v*100:.1f}%" for k, v in probs.items()},
        "odds": odds,
        "xg": f"{lh:.2f} – {la:.2f}"
    }

# Arayüz
st.title("📅 Günlük Fikstür Tahmin Aracı")

today = datetime.now().date()
dates = {
    "Bugün": today.strftime("%Y-%m-%d"),
    "Yarın": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
    "Sonraki Gün": (today + timedelta(days=2)).strftime("%Y-%m-%d")
}
selected_label = st.sidebar.selectbox("Tarih Seç", list(dates.keys()))
selected_date = dates[selected_label]

fixtures = get_fixtures(selected_date)

if not fixtures:
    st.warning(f"{selected_label} ({selected_date}) için maç verisi yok veya API yavaş. Başka tarih dene.")
else:
    st.subheader(f"{selected_label} ({selected_date}) – {len(fixtures)} maç listeleniyor")

    if st.button("🔥 En Yüksek Üst 2.5 Olasılıklı Maçlar (Top 10)", type="primary"):
        with st.spinner("Tüm maçlar hesaplanıyor... (yavaş olabilir)"):
            high_prob = []
            for event in fixtures:
                if not isinstance(event, dict):
                    continue
                status = event.get("status") or {}
                if status.get("type") in ["finished", "canceled"]:
                    continue
                home = event.get("homeTeam", {})
                away = event.get("awayTeam", {})
                if not home.get("id") or not away.get("id"):
                    continue
                pred = predict_match(home["id"], away["id"])
                if pred:
                    over_pct_str = pred["probs"]["over_2.5"]
                    try:
                        pct = float(over_pct_str.replace("%", ""))
                        high_prob.append({
                            "saat": datetime.fromtimestamp(event.get("startTimestamp", 0)).strftime("%H:%M"),
                            "lig": event.get("tournament", {}).get("name", "—"),
                            "mac": f"{home.get('name', '?')} - {away.get('name', '?')}",
                            "over_2.5": over_pct_str,
                            "pct": pct
                        })
                    except:
                        pass
            if high_prob:
                high_prob.sort(key=lambda x: x["pct"], reverse=True)
                for item in high_prob[:10]:
                    st.write(f"**{item['saat']}** | {item['lig']} | {item['mac']} → **Üst 2.5: {item['over_2.5']}**")
            else:
                st.info("Yeterli veri yok veya maçlar başlamamış.")

    # Fikstür listesi
    for event in fixtures:
        if not isinstance(event, dict):
            continue
        status = event.get("status") or {}
        if status.get("type") in ["finished", "canceled"]:
            continue
        home = event.get("homeTeam", {})
        away = event.get("awayTeam", {})
        saat = datetime.fromtimestamp(event.get("startTimestamp", 0)).strftime("%H:%M")
        lig = event.get("tournament", {}).get("name", "—")

        expander_label = f"⏰ {saat} | {lig} | {home.get('name', '?')} - {away.get('name', '?')}"
        with st.expander(expander_label):
            if st.button("Tahmin Hesapla", key=f"tahmin_{event.get('id', 'no_id')}"):
                with st.spinner("Tahmin hesaplanıyor..."):
                    pred = predict_match(home.get("id"), away.get("id"))
                    if pred:
                        st.write(f"**Beklenen Gol (xG)**: {pred['xg']}")
                        st.write("**Maç Sonucu**")
                        for r in ["1", "X", "2"]:
                            st.write(f"{r}: {pred['probs'][r]} → oran ≈ {pred['odds'][r]}")
                        st.write("**Üst 2.5**")
                        st.write(f"{pred['probs']['over_2.5']} → oran ≈ {pred['odds']['over_2.5']}")
                    else:
                        st.warning("Yeterli son maç verisi yok (en az 3 maç lazım).")

st.caption("Not: Tahminler sadece son 6 maç istatistiğine dayalı basit Poisson modeli. Bahis için ek faktörleri (sakatlık, motivasyon) de düşün.")
