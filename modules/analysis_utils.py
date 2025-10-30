import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

# 테스트 사용자 ID
TEMP_USER_ID = "61720be3-19c6-4383-8563-85a6f2d4e795"

def _process_raw_data(data: list):
    if not data:
        return pd.DataFrame()

    records = []
    for item in data:
        report_info = item.get('test_reports_qsin')
        if report_info and report_info.get('snr_level') is not None:
            records.append({
                'session_id': report_info.get('session_id'),
                'user_id': report_info.get('user_id'),
                'sentence_id': item['index'],
                'sentences': item['sentences'],
                'total_score': item['total_score'],
                'full_sentence': item['full_sentence'],
                'score': item['score'],
                'snr_level': report_info['snr_level']
            })
    
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['num_keywords'] = df['sentences'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df = df[df['num_keywords'] > 0]
    df['correct_rate'] = df['total_score'] / df['num_keywords']
    
    return df

@st.cache_data(ttl=600)
def get_all_sentence_data(_supabase_client, use_dummy_prefix: bool, sentence_id: int | None = None):
    try:
        page_size = 1000  # Supabase 기본 페이지 크기

        base_select = (
            "index, sentences, total_score, full_sentence, score,"
            " test_reports_qsin!inner(snr_level, session_id, user_id)"
        )

        def add_sentence_filter(q):
            return q.eq('index', sentence_id) if sentence_id is not None else q

        def fetch_all_with_filter(apply_filter_fn):            
            all_rows = []
            page = 0
            while True:
                start_index = page * page_size
                end_index = start_index + page_size - 1

                query = _supabase_client.table('score_qsin').select(base_select)
                query = add_sentence_filter(query)
                query = apply_filter_fn(query)

                resp = query.range(start_index, end_index).execute()
                rows = resp.data or []

                if not rows:
                    break

                all_rows.extend(rows)

                if len(rows) < page_size:
                    break

                page += 1

            return all_rows

        if use_dummy_prefix:
            # dummy_ 접두사 세션 또는 특정 테스트 사용자 데이터 포함
            r1 = fetch_all_with_filter(
                lambda q: q.ilike('test_reports_qsin.session_id', 'dummy_%')
            )

            r2 = fetch_all_with_filter(
                lambda q: q.eq('test_reports_qsin.user_id', TEMP_USER_ID)
            )

            dedup = {}
            for row in r1 + r2:
                rep = row.get('test_reports_qsin') or {}
                key = (row.get('index'), rep.get('session_id'), rep.get('snr_level'))
                if key not in dedup:
                    dedup[key] = row

            all_records = list(dedup.values())

        else:
            all_records = fetch_all_with_filter(
                lambda q: q.not_.ilike('test_reports_qsin.session_id', 'dummy_%')
                            .neq('test_reports_qsin.user_id', TEMP_USER_ID)
            )

        df = _process_raw_data(all_records)
        if df.empty:
            return pd.DataFrame()

        # 반환 컬럼 정리
        return df[['session_id', 'user_id', 'sentence_id', 'full_sentence',
                   'snr_level', 'score', 'total_score', 'correct_rate']]

    except Exception as e:
        st.error(f"전체 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

def estimate_snr50_for_sentence(data: pd.DataFrame):
    """
    단일 문장에 대한 데이터(snr_level, correct_rate)를 받아
    로지스틱 회귀 분석으로 SNR-50과 기울기를 추정합니다.
    """
    agg_data = data.groupby('snr_level')['correct_rate'].mean().reset_index()

    if len(agg_data) < 3:
        return {'status': 'Error: Not Enough Data Points', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}
    
    if agg_data['correct_rate'].min() == agg_data['correct_rate'].max():
        return {'status': 'Error: All Same Results', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}

    X = agg_data[['snr_level']]
    y = agg_data['correct_rate']
    
    try:
        # Deterministic resampling instead of random binomial to ensure reproducibility
        n_trials = 100
        X_resampled = np.repeat(agg_data['snr_level'].values, n_trials).reshape(-1, 1)
        y_resampled = []
        for rate in agg_data['correct_rate']:
            successes = int(round(rate * n_trials))
            failures = n_trials - successes
            y_resampled.extend([1] * successes + [0] * failures)

        model = LogisticRegression(solver='liblinear')
        model.fit(X_resampled, y_resampled)

        intercept = model.intercept_[0]
        coef = model.coef_[0][0]

        if abs(coef) < 1e-6:
            return {'status': 'Error: Zero Slope', 'snr_50': None, 'slope': None, 'model': model, 'plot_data': agg_data}

        snr_50 = -intercept / coef
        slope = coef / 4 * 100

        # 신뢰도 체크: 추정된 SNR-50이 테스트 범위를 너무 많이 벗어나는지 확인
        snr_min, snr_max = X['snr_level'].min(), X['snr_level'].max()
        # 허용 범위를 테스트된 SNR 범위보다 5dB 더 넓게 설정
        valid_range_min = snr_min - 5
        valid_range_max = snr_max + 5
        
        # 신뢰도 및 난이도 등급 체크
        snr_min, snr_max = X['snr_level'].min(), X['snr_level'].max()
        valid_range_min = snr_min - 5
        valid_range_max = snr_max + 5
        
        if not (valid_range_min <= snr_50 <= valid_range_max):
            validity = 'Extrapolated'
        else:
            if 0.5 <= snr_50 <= 3.5:
                validity = 'Ideal'
            elif -1.0 <= snr_50 <= 5.0:
                validity = 'Acceptable'
            else:
                validity = 'Warning'

        return {
            'status': 'Success', 
            'snr_50': float(snr_50), 
            'slope': float(slope),
            'validity': validity,
            'model': model, 
            'plot_data': agg_data
        }

    except Exception as e:
        return {'status': f'Error: {e}', 'snr_50': None, 'slope': None, 'validity': 'Error', 'model': None, 'plot_data': agg_data}

def analyze_all_sentences(data: pd.DataFrame):
    """
    전체 데이터에 대해 문장별로 SNR-50과 기울기를 분석합니다.
    """
    if data.empty:
        return pd.DataFrame()
    
    results = []
    sentence_ids = data['sentence_id'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sentence_id in enumerate(sentence_ids):
        status_text.text(f"문장 {sentence_id}번 분석 중... ({i+1}/{len(sentence_ids)})")
        progress_bar.progress((i + 1) / len(sentence_ids))
        
        sentence_data = data[data['sentence_id'] == sentence_id]
        result = estimate_snr50_for_sentence(sentence_data)
        
        if result['status'] == 'Success':
            full_sentence = sentence_data['full_sentence'].iloc[0]
            
            # 총 점수와 평균 점수 계산
            total_score_sum = sentence_data['total_score'].sum()
            avg_score = sentence_data['total_score'].mean()
            
            results.append({
                'sentence_id': sentence_id,
                'full_sentence': full_sentence,
                'snr_50': result['snr_50'],
                'slope': result['slope'],
                'validity': result.get('validity', 'N/A'),
                'total_score_sum': total_score_sum,
                'avg_score': avg_score,
                'data_points': len(sentence_data),
                'snr_levels': len(sentence_data['snr_level'].unique())
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def display_analysis_metrics(snr50_val, slope_val):
    """
    분석 결과 메트릭을 표시하는 공통 함수
    """
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="추정 SNR-50",
                value=f"{snr50_val:.2f} dB",
                help="피험자가 이 문장의 단어를 50% 확률로 맞추는 데 필요한 신호 대 잡음비(Signal-to-Noise Ratio)입니다. **값이 낮을수록 더 시끄러운 환경에서도 잘 들리는 쉬운 문장**임을 의미합니다."
            )
        with col2:
            st.metric(
                label="기울기",
                value=f"{slope_val:.2f} %/dB",
                help="SNR-50 지점 부근에서 SNR이 1dB 변할 때마다 정답률이 몇 %씩 변하는지를 나타내는 **민감도 지표**입니다. **값이 높을수록 소음 변화에 따라 난이도가 급격하게 변하는 문장**임을 의미합니다. 기울기가 높은 문장일수록 신뢰도 높은 측정이 가능합니다."
            )

def create_psychometric_plot(processed_data, result, sentence_id=None, title_suffix=""):
    """
    Psychometric Function Curve를 생성하는 공통 함수
    """
    if processed_data.empty:
        st.warning("시각화할 데이터가 없습니다.")
        return None
    
    model = result.get('model')
    snr50_val = result.get('snr_50')
    plot_data = result.get('plot_data')
    validity = result.get('validity', 'Good')
    
    fig = go.Figure()
    
    # 1. Box Plot으로 전체 데이터 분포 표시
    fig.add_trace(go.Box(
        x=processed_data['snr_level'],
        y=processed_data['correct_rate'],
        name='정답률 분포',
        boxpoints=False,
        marker_color='orange',
        boxmean=True,
        visible='legendonly'
    ))

    # 2. Scatter Plot으로 평균 정답률 표시
    if not plot_data.empty:
        fig.add_trace(go.Scatter(
            x=plot_data['snr_level'],
            y=plot_data['correct_rate'],
            mode='lines+markers',
            name='평균 정답률',
            line=dict(color='dodgerblue', dash='dot'),
            marker=dict(size=10, color='dodgerblue', symbol='circle')
        ))

    # 3. 로지스틱 회귀 곡선 및 SNR-50 라인 표시
    if model and snr50_val is not None:
        agg_plot_data = processed_data.groupby('snr_level')['correct_rate'].mean().reset_index()
        x_range = np.linspace(agg_plot_data['snr_level'].min() - 5, agg_plot_data['snr_level'].max() + 5, 100)
        y_curve = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
        
        fig.add_trace(go.Scatter(
            x=x_range, y=y_curve, mode='lines', name='로지스틱 회귀 곡선', line=dict(color='red', width=2)
        ))

        # Validity에 따라 SNR-50 라인 스타일 변경
        if validity == 'Extrapolated':
            line_color = "grey"
            annotation_text = f"SNR-50: {snr50_val:.2f} dB (추정치)"
            line_dash = "dot"
        else:
            line_color = "green"
            annotation_text = f"SNR-50: {snr50_val:.2f} dB"
            line_dash = "dash"

        fig.add_vline(x=snr50_val, line_width=2, line_dash=line_dash, line_color=line_color,
                    annotation_text=annotation_text, annotation_position="top right")
        fig.add_hline(y=0.5, line_width=2, line_dash="dash", line_color="green")
    
    # 제목 설정
    if sentence_id:
        full_sentence_text = processed_data['full_sentence'].iloc[0]
        title = f"문장 {sentence_id}번 : \"{full_sentence_text}\""
    else:
        title = f"전체 데이터 분석 결과{title_suffix}"
    
    fig.update_layout(
        title=title,
        xaxis_title="SNR Level (dB)",
        yaxis_title="정답률 (Correct Rate)",
        yaxis_range=[-0.05, 1.05],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_combined_psychometric_plot(
    sentence_ids: list[int],
    include_logistic: bool = True,
    include_mean: bool = False,
    show_legend: bool = False,
    precomputed_results: pd.DataFrame | None = None,
    snr_range: tuple[float, float] = (-10, 10),
):
    """
    여러 문장(sentence_id)의 심리물리 곡선을 하나의 Figure에 겹쳐서 표시합니다.

    - sentence_ids: 겹쳐서 표시할 문장 ID 목록
    - include_logistic: 각 문장에 대해 로지스틱 회귀 곡선을 함께 표시할지 여부
    - precomputed_results: 사전 계산된 분석 결과 (snr_50, slope, validity 포함)
    - snr_range: SNR 데이터 범위 (min, max)
    """
    if not sentence_ids:
        st.warning("시각화할 데이터가 없습니다.")
        return None

    fig = go.Figure()

    # 전역 x 구간 설정
    global_min, global_max = snr_range
    x_range_global = np.linspace(global_min - 5, global_max + 5, 100)

    # 표(사전 계산) 결과를 빠르게 찾기 위한 맵 구성
    results_map = {}
    if precomputed_results is not None and not precomputed_results.empty:
        # 기대 컬럼: sentence_id, snr_50, slope, validity
        for _, row in precomputed_results.iterrows():
            sid = int(row['sentence_id']) if not pd.isna(row['sentence_id']) else None
            if sid is None:
                continue
            results_map[sid] = {
                'snr_50': row.get('snr_50', None),
                'slope': row.get('slope', None),  # %/dB
                'validity': row.get('validity', 'Good'),
            }

    # 선택한 문장마다 로지스틱 곡선 추가
    for sid in sentence_ids:
        if include_logistic:
            # 표의 snr_50, slope를 그대로 사용해 로지스틱 곡선을 복원 (재학습 없음)
            r = results_map.get(sid)
            if r is not None and r.get('snr_50') is not None and r.get('slope') is not None:
                snr50 = float(r['snr_50'])
                slope_pct_per_db = float(r['slope'])  # %/dB
                validity = r.get('validity', 'Warning')
                
                # Validity와 SNR-50 값에 따른 곡선 색상 결정 (gradient 적용)
                if validity == 'Ideal':
                    # 녹색 계열: 0.5~3.5 범위에서 연두색 → 진한 녹색
                    ratio = (snr50 - 0.5) / 3.0  # 0.0 ~ 1.0
                    ratio = max(0.0, min(1.0, ratio))
                    r_val = int(144 * (1 - ratio))  # 144 → 0
                    g_val = int(238 - 110 * ratio)  # 238 → 128
                    b_val = int(144 * (1 - ratio))  # 144 → 0
                    curve_color = f'rgba({r_val}, {g_val}, {b_val}, 0.6)'
                elif validity == 'Acceptable':
                    # 주황색 계열: -1.0~5.0 범위에서 노란색 → 주황색 → 갈색
                    ratio = (snr50 - (-1.0)) / 6.0  # 0.0 ~ 1.0
                    ratio = max(0.0, min(1.0, ratio))
                    r_val = 255
                    g_val = int(215 - 100 * ratio)  # 215 → 115
                    b_val = int(100 * (1 - ratio))  # 100 → 0
                    curve_color = f'rgba({r_val}, {g_val}, {b_val}, 0.6)'
                elif validity == 'Warning':
                    # 빨간색 계열: 분산도에 따라 밝은 빨강 → 진한 빨강
                    distance = abs(snr50 - 2.0)  # 2dB에서 얼마나 멀리 떨어져 있는지
                    ratio = min(1.0, distance / 10.0)  # 거리가 멀수록 진하게
                    r_val = 255
                    g_val = int(100 * (1 - ratio))  # 100 → 0
                    b_val = int(100 * (1 - ratio))  # 100 → 0
                    curve_color = f'rgba({r_val}, {g_val}, {b_val}, 0.6)'
                else:  # Extrapolated
                    # 회색 계열: 데이터 범위를 벗어난 정도에 따라 밝은 회색 → 진한 회색
                    distance = abs(snr50 - 2.0)
                    ratio = min(1.0, distance / 15.0)
                    gray_val = int(180 - 60 * ratio)  # 180 → 120
                    curve_color = f'rgba({gray_val}, {gray_val}, {gray_val}, 0.4)'
                
                coef = (slope_pct_per_db / 100.0) * 4.0  # 로지스틱 기울기와의 변환
                # p(x) = 1 / (1 + exp(-(a + b*x)))
                # snr_50 = -a/b => a = -b*snr_50
                a = -coef * snr50
                y_curve = 1.0 / (1.0 + np.exp(-(a + coef * x_range_global)))

                fig.add_trace(go.Scatter(
                    x=x_range_global,
                    y=y_curve,
                    mode='lines',
                    name=f"문장 {sid} 로지스틱",
                    line=dict(width=1.5, color=curve_color, dash='dot'),
                    showlegend=show_legend
                ))

                # SNR-50 포인트 표시 (y는 0.5)
                validity = r.get('validity', 'Warning')
                
                # 유효성 및 2dB 근접성에 따라 마커 색상 결정
                color_map = {
                    'Ideal': 'green',
                    'Acceptable': 'orange',
                    'Warning': 'red',
                    'Extrapolated': 'grey'
                }
                marker_color = color_map.get(validity, 'purple') # 기본값

                hover_text = (
                    f"문장 {sid} SNR-50: %{{x:.2f}} dB<br>"
                    f"기울기: {slope_pct_per_db:.2f} %/dB<br>"
                    f"등급: {validity}<extra></extra>"
                )

                fig.add_trace(go.Scatter(
                    x=[snr50],
                    y=[0.5],
                    mode='markers',
                    name=f"문장 {sid} SNR-50",
                    marker=dict(symbol='x', size=9, color=marker_color),
                    showlegend=False,
                    hovertemplate=hover_text
                ))

    title = (
        "전체 문장들에 대한 Psychometric Function"
    )

    fig.update_layout(
        title=title,
        xaxis_title="SNR Level (dB)",
        yaxis_title="정답률 (Correct Rate)",
        yaxis_range=[-0.05, 1.05],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest'
    )
    
    # WebGL 렌더링 활성화 (대량 데이터 성능 최적화)
    fig.update_traces(line=dict(width=1.5), selector=dict(mode='lines'))
    
    return fig
