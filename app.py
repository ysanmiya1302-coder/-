import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import urllib.request
import time

# 画面設定（スマホ表示の最適化）
st.set_page_config(page_title="中央競馬 AI予測", page_icon="🏇", layout="wide")

st.title("🏇 中央競馬 AI予測ダッシュボード")

# 競馬場マップ
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉'
}

def get_race_title(race_id):
    venue_code = race_id[4:6]
    race_num = int(race_id[10:12])
    venue_name = VENUE_MAP.get(venue_code, '競馬場')
    return f"{venue_name} {race_num}R"

# モデル & データの読み込み
@st.cache_resource
def load_assets():
    model_win = joblib.load('model_win.pkl')
    model_place = joblib.load('model_place.pkl')
    config = joblib.load('model_config.pkl')
    df = pd.read_parquet('history_df.parquet')
    return model_win, model_place, config['features'], config['cat_cols'], df

try:
    model_win, model_place, features, cat_cols, df = load_assets()
    st.sidebar.success("✅ AIモデル読み込み完了")
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# スマホ向け操作パネル
st.sidebar.header("🎯 予測設定")
prefix_input = st.sidebar.text_input("10桁ベースID", value="2026090408", help="例: 9/26 阪神1RのURL頭10桁")

if st.sidebar.button("🚀 全12レースをAI予測", type="primary"):
    RACE_IDS = [f"{prefix_input}{r:02d}" for r in range(1, 13)]
    all_predictions = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, race_id in enumerate(RACE_IDS):
        race_title = get_race_title(race_id)
        progress_text.text(f"⏳ 解析中: 【{race_title}】 ({idx+1}/12)")
        
        try:
            url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            html_bytes = urllib.request.urlopen(req).read()
            tables = pd.read_html(html_bytes)
            shutuba_df = tables[0]
            shutuba_df.columns = [re.sub(r'\s+', '', str(c)) for c in shutuba_df.columns]

            horse_col = [c for c in shutuba_df.columns if '馬名' in c][0]
            jockey_col = [c for c in shutuba_df.columns if '騎手' in c][0]
            trainer_col = [c for c in shutuba_df.columns if '調教師' in c or '厩舎' in c][0]
            num_col = [c for c in shutuba_df.columns if '馬番' in c or '頭番' in c]
            
            shutuba_df['馬番_clean'] = pd.to_numeric(shutuba_df[num_col[0]], errors='coerce') if num_col else np.arange(1, len(shutuba_df) + 1)
            shutuba_df['馬名_clean'] = shutuba_df[horse_col].apply(lambda x: re.sub(r'[^\w]', '', str(x)) if pd.notna(x) else '')
            shutuba_df['騎手_clean'] = shutuba_df[jockey_col].astype(str).str.replace(r'\s+', '', regex=True)
            shutuba_df['調教師_clean'] = shutuba_df[trainer_col].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip()
            shutuba_df['枠番_num'] = pd.to_numeric(shutuba_df.get('枠', shutuba_df.get('枠番', np.nan)), errors='coerce')
            shutuba_df['斤量_num'] = pd.to_numeric(shutuba_df.get('斤量', np.nan), errors='coerce')

            weight_col = [c for c in shutuba_df.columns if '馬体重' in c]
            if weight_col:
                shutuba_df['馬体重_数値'] = shutuba_df[weight_col[0]].astype(str).str.extract(r'(\d+)').astype(float)
            else:
                shutuba_df['馬体重_数値'] = np.nan

            latest_records = []
            for horse in shutuba_df['馬名_clean']:
                horse_history = df[df['馬名'] == horse].sort_values(by='レースID', ascending=False)
                if len(horse_history) > 0:
                    last_race = horse_history.iloc[0]
                    latest_records.append({
                        '馬名_clean': horse,
                        '前走タイム差': last_race.get('1着とのタイム差', np.nan),
                        '前走着順': last_race.get('着順_num', np.nan),
                        '前走上がり': last_race.get('上がり_秒', np.nan),
                        '前走上がり順位': last_race.get('上がり順位', np.nan),
                        '前走最終コーナー通過': last_race.get('最終コーナー通過', np.nan),
                        '前走距離': last_race.get('距離_num', np.nan),
                        '前走コース': last_race.get('コース種別', '不明'),
                        '前走からの間隔': 2.0
                    })
                else:
                    latest_records.append({
                        '馬名_clean': horse,
                        '前走タイム差': np.nan, '前走着順': np.nan, '前走上がり': np.nan,
                        '前走上がり順位': np.nan, '前走最終コーナー通過': np.nan,
                        '前走距離': np.nan, '前走コース': '不明', '前走からの間隔': np.nan
                    })

            history_df = pd.DataFrame(latest_records)
            input_df = pd.merge(shutuba_df, history_df, on='馬名_clean', how='left')

            input_df['距離_num'] = 2000.0 if '距離_num' not in input_df.columns else input_df['距離_num']
            input_df['距離変化'] = input_df['距離_num'] - input_df['前走距離']
            input_df['距離区分'] = '中距離'
            input_df['コース種別'] = '芝'
            input_df['馬場状態'] = '良'
            input_df['コース変化'] = input_df['コース種別'].astype(str) + '←' + input_df['前走コース'].astype(str)
            input_df['コース_距離'] = input_df['コース種別'].astype(str) + '_' + input_df['距離区分'].astype(str)
            input_df['コース_距離_枠'] = input_df['コース_距離'].astype(str) + '_枠' + input_df['枠番_num'].astype(str)

            for col in features:
                if col not in input_df.columns:
                    input_df[col] = np.nan

            X_input = input_df[features].copy()
            for col in X_input.columns:
                if col in cat_cols or X_input[col].dtype == 'object':
                    X_input[col] = X_input[col].astype(str).astype('category')

            pred_place = model_place.predict_proba(X_input)[:, 1]
            pred_win = model_win.predict_proba(X_input)[:, 1]

            input_df['レースID'] = race_id
            input_df['レース名'] = race_title
            input_df['予測勝率'] = pred_win
            input_df['勝率_pct'] = (input_df['予測勝率'] / input_df['予測勝率'].sum() * 100).round(1)
            input_df['複勝率_pct'] = (pred_place * 100).round(1)
            input_df['AI順位'] = input_df['複勝率_pct'].rank(ascending=False, method='first').astype(int)

            mark_map = {1: '◎', 2: '◯', 3: '▲', 4: '△', 5: '△'}
            input_df['印'] = input_df['AI順位'].map(lambda x: mark_map.get(x, ''))

            all_predictions.append(input_df)
        except Exception as e:
            pass
            
        progress_bar.progress((idx + 1) / 12)
        time.sleep(0.1)

    progress_text.text("🎉 解析完了！")

    if all_predictions:
        total_df = pd.concat(all_predictions, ignore_index=True)
        
        # 🌟 自信度Top5表示
        st.subheader("🔥 AI厳選・自信度ランキング Top5")
        honmei = total_df[total_df['AI順位'] == 1].sort_values(by='複勝率_pct', ascending=False).head(5)
        for _, row in honmei.iterrows():
            st.success(f"✨ **【{row['レース名']}】** 本命: **◎ {row['馬名_clean']}** (騎手: {row['騎手_clean']}) | 予測複勝率: 🔥 **{row['複勝率_pct']}%** | 勝率: {row['勝率_pct']}%")

        # 📋 各レース結果のタブ表示
        st.subheader("📋 レース別予想結果 & 買い目")
        r_ids = total_df['レースID'].unique()
        tab_names = [total_df[total_df['レースID'] == r]['レース名'].iloc[0] for r in r_ids]
        tabs = st.tabs(tab_names)
        
        for tab, r_id in zip(tabs, r_ids):
            with tab:
                r_df = total_df[total_df['レースID'] == r_id].sort_values(by='AI順位')
                disp = r_df[['印', 'AI順位', '馬番_clean', '馬名_clean', '騎手_clean', '勝率_pct', '複勝率_pct']].rename(
                    columns={'馬番_clean': '馬番', '馬名_clean': '馬名', '騎手_clean': '騎手'}
                )
                st.dataframe(disp, use_container_width=True)

                top1 = r_df[r_df['AI順位'] == 1].iloc[0]
                opponents = r_df[r_df['AI順位'].isin([2, 3, 4, 5])]
                aite_names = [f"{row['印']}{row['馬名_clean']}" for _, row in opponents.iterrows()]

                st.info(f"🎯 **推奨買い目**: ◎ **{top1['馬名_clean']}** から 馬連/ワイド流し ➔ {' / '.join(aite_names)}")
