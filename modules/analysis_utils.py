import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

# 테스트 사용자 ID
TEMP_USER_ID = "61720be3-19c6-4383-8563-85a6f2d4e795"

# K-Quick-SIN 정상 규준
TARGET_SNR50 = -4.16  # dB
ABSOLUTE_TOLERANCE = 2.0  # +-2.0 dB
SLOPE_THRESHOLD = 3.0  # %/dB 미만 시 변별력 부족
SENTENCE_SD_THRESHOLD = 3.0  # dB, 문장 간 SD 상한

# SNR Loss 등급 기준 (Killion, 1997 참고)
SNR_LOSS_GRADES = [
    ('정상', 0, 3),
    ('경도', 3, 7),
    ('중도', 7, 15),
    ('고도', 15, float('inf'))
]

def _process_raw_data(data: list):
    if not data:
        return pd.DataFrame()

    records = []
    for item in data:
        report_info = item.get('test_reports_qsin')
        if report_info and report_info.get('snr_level') is not None:
            # patient_user 테이블에서 hearing_loss 추출
            patient_info = report_info.get('patient_user')
            hearing_loss = None
            if isinstance(patient_info, dict):
                hearing_loss = patient_info.get('hearing_loss')

            records.append({
                'session_id': report_info.get('session_id'),
                'user_id': report_info.get('user_id'),
                'patient_user_id': report_info.get('patient_user_id'),
                'hearing_loss': hearing_loss,
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
        page_size = 1000

        base_select = (
            "index, sentences, total_score, full_sentence, score,"
            " test_reports_qsin!inner(snr_level, session_id, user_id, patient_user_id,"
            " patient_user(hearing_loss))"
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

        return df[['session_id', 'user_id', 'patient_user_id', 'hearing_loss',
                   'sentence_id', 'full_sentence', 'snr_level', 'score',
                   'total_score', 'correct_rate']]

    except Exception as e:
        st.error(f"전체 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

def estimate_snr50_for_sentence(
    data: pd.DataFrame
):
    """
    단일 문장에 대한 데이터(snr_level, correct_rate)를 받아
    로지스틱 회귀 분석으로 SNR-50과 기울기를 추정합니다.

    이 함수는 순수하게 수치적 분석(SNR-50, Slope)만 수행하며,
    등급(Validity) 분류는 수행하지 않습니다. (단, 데이터 범위 부족으로 인한 Extrapolated 여부는 판단)
    """
    agg_data = data.groupby('snr_level')['correct_rate'].mean().reset_index()

    if len(agg_data) < 3:
        return {'status': 'Error: Not Enough Data Points', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}

    if agg_data['correct_rate'].min() == agg_data['correct_rate'].max():
        return {'status': 'Error: All Same Results', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}

    X = agg_data[['snr_level']]
    y = agg_data['correct_rate']

    try:
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

        snr_min, snr_max = X['snr_level'].min(), X['snr_level'].max()
        valid_range_min = snr_min - 5
        valid_range_max = snr_max + 5

        if not (valid_range_min <= snr_50 <= valid_range_max):
            validity = 'Extrapolated'
        else:
            validity = 'Analyzed'

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

def calculate_dynamic_ranges(analysis_results_df: pd.DataFrame) -> dict:
    """
    분석 결과 데이터프레임에서 유효한 SNR-50 데이터를 추출하여
    IQR 기반 및 Mean/Std 기반의 동적 범위를 계산합니다.
    """
    valid_snr_data = analysis_results_df[analysis_results_df['validity'] != 'Extrapolated']['snr_50']

    if not valid_snr_data.empty and len(valid_snr_data) > 1:
        stats = valid_snr_data.describe()
        mean = stats['mean']
        std = stats['std']
        q1 = stats['25%']
        q3 = stats['75%']
        iqr = q3 - q1

        return {
            'iqr': {
                'ideal': (q1, q3),
                'acceptable': (q1 - 0.5 * iqr, q3 + 0.5 * iqr)
            },
            'mean_std': {
                'ideal': (mean - 0.5 * std, mean + 0.5 * std),
                'acceptable': (mean - 1.0 * std, mean + 1.0 * std)
            }
        }
    else:
        return {
            'iqr': {'ideal': (-8.56, -5.57), 'acceptable': (-10.05, -4.07)},
            'mean_std': {'ideal': (-8.13, -5.98), 'acceptable': (-9.21, -4.90)}
        }

def reclassify_with_absolute_criterion(
    df: pd.DataFrame,
    target: float = TARGET_SNR50,
    tolerance: float = ABSOLUTE_TOLERANCE,
    ideal_range: tuple[float, float] | None = None,
    acceptable_range: tuple[float, float] | None = None
) -> pd.DataFrame:
    """
    1차 절대 기준(Absolute Criterion) + 2차 상대 기준(Relative Sub-grade)으로 분류합니다.

    - clinical_validity: Pass / Fail / Extrapolated
    - sub_grade: Ideal / Acceptable / Marginal / - (Fail 또는 Extrapolated인 경우)
    """
    if df.empty:
        return df

    df_copy = df.copy()

    def _classify(row):
        if row.get('validity') == 'Extrapolated':
            return 'Extrapolated', '-'

        snr = row.get('snr_50')
        if snr is None:
            return 'Error', '-'

        if abs(snr - target) <= tolerance:
            clinical = 'Pass'
            if ideal_range and acceptable_range:
                if ideal_range[0] <= snr <= ideal_range[1]:
                    sub = 'Ideal'
                elif acceptable_range[0] <= snr <= acceptable_range[1]:
                    sub = 'Acceptable'
                else:
                    sub = 'Marginal'
            else:
                sub = '-'
        else:
            clinical = 'Fail'
            sub = '-'

        return clinical, sub

    results = df_copy.apply(_classify, axis=1, result_type='expand')
    df_copy['clinical_validity'] = results[0]
    df_copy['sub_grade'] = results[1]
    return df_copy

def classify_snr_loss_grade(snr_loss: float) -> str:
    """SNR Loss 값에 따른 청력 손실 등급 분류 (음수 = 정상보다 좋음 → 정상)"""
    if snr_loss <= 0:
        return '정상'
    for grade, low, high in SNR_LOSS_GRADES:
        if low <= snr_loss < high:
            return grade
    return '고도'

def analyze_all_sentences(data: pd.DataFrame, label: str = ""):
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
        status_text.text(f"{label}문장 {sentence_id}번 분석 중... ({i+1}/{len(sentence_ids)})")
        progress_bar.progress((i + 1) / len(sentence_ids))

        sentence_data = data[data['sentence_id'] == sentence_id]
        result = estimate_snr50_for_sentence(sentence_data)

        if result['status'] == 'Success':
            full_sentence = sentence_data['full_sentence'].iloc[0]
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

def analyze_subjects(data: pd.DataFrame, target: float = TARGET_SNR50):
    """
    난청군 피험자별 SNR-50 및 SNR Loss를 분석합니다.
    각 피험자의 전체 문장 데이터를 풀링하여 SNR-50을 추정하고,
    목표값과의 차이(SNR Loss)로 청력 손실 등급을 분류합니다.
    """
    if data.empty:
        return pd.DataFrame()

    results = []
    subjects = data['patient_user_id'].dropna().unique()

    if len(subjects) == 0:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, subject_id in enumerate(subjects):
        status_text.text(f"피험자 {i+1}/{len(subjects)} 분석 중...")
        progress_bar.progress((i + 1) / len(subjects))

        subject_data = data[data['patient_user_id'] == subject_id]
        result = estimate_snr50_for_sentence(subject_data)

        if result['status'] == 'Success':
            snr_50 = result['snr_50']
            validity = result.get('validity', 'Analyzed')
            is_extrapolated = validity == 'Extrapolated'

            snr_loss = snr_50 - target
            grade = None if is_extrapolated else classify_snr_loss_grade(snr_loss)

            results.append({
                'patient_user_id': subject_id,
                'snr_50': snr_50,
                'snr_loss': snr_loss,
                'grade': grade,
                'validity': validity,
                'slope': result['slope'],
                'data_points': len(subject_data),
                'sentences_tested': len(subject_data['sentence_id'].unique())
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

    fig.add_trace(go.Box(
        x=processed_data['snr_level'],
        y=processed_data['correct_rate'],
        name='정답률 분포',
        boxpoints=False,
        marker_color='orange',
        boxmean=True,
        visible='legendonly'
    ))

    if not plot_data.empty:
        fig.add_trace(go.Scatter(
            x=plot_data['snr_level'],
            y=plot_data['correct_rate'],
            mode='lines+markers',
            name='평균 정답률',
            line=dict(color='dodgerblue', dash='dot'),
            marker=dict(size=10, color='dodgerblue', symbol='circle')
        ))

    if model and snr50_val is not None:
        agg_plot_data = processed_data.groupby('snr_level')['correct_rate'].mean().reset_index()
        x_range = np.linspace(agg_plot_data['snr_level'].min() - 5, agg_plot_data['snr_level'].max() + 5, 100)
        y_curve = model.predict_proba(x_range.reshape(-1, 1))[:, 1]

        fig.add_trace(go.Scatter(
            x=x_range, y=y_curve, mode='lines', name='로지스틱 회귀 곡선', line=dict(color='red', width=2)
        ))

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
    ideal_range: tuple[float, float] = (-8.56, -5.57),
    acceptable_range: tuple[float, float] = (-10.05, -4.07)
):
    """
    여러 문장(sentence_id)의 심리물리 곡선을 하나의 Figure에 겹쳐서 표시합니다.
    """
    if not sentence_ids:
        st.warning("시각화할 데이터가 없습니다.")
        return None

    fig = go.Figure()

    global_min, global_max = snr_range
    x_range_global = np.linspace(global_min - 5, global_max + 5, 100)

    results_map = {}
    if precomputed_results is not None and not precomputed_results.empty:
        for _, row in precomputed_results.iterrows():
            sid = int(row['sentence_id']) if not pd.isna(row['sentence_id']) else None
            if sid is None:
                continue
            results_map[sid] = {
                'snr_50': row.get('snr_50', None),
                'slope': row.get('slope', None),
                'validity': row.get('validity', 'Good'),
            }

    for sid in sentence_ids:
        if include_logistic:
            r = results_map.get(sid)
            if r is not None and r.get('snr_50') is not None and r.get('slope') is not None:
                validity = r.get('validity', 'Warning')

                if validity == 'Extrapolated':
                    continue

                snr50 = float(r['snr_50'])
                slope_pct_per_db = float(r['slope'])

                if validity == 'Ideal':
                    span = ideal_range[1] - ideal_range[0]
                    normalized_pos = (snr50 - ideal_range[0]) / span if span > 0 else 0.5
                    normalized_pos = max(0.0, min(1.0, normalized_pos))
                    color_val = int(180 * (1 - normalized_pos))
                    curve_color = f'hsl({color_val}, 70%, 45%)'
                elif validity == 'Acceptable':
                    span = acceptable_range[1] - acceptable_range[0]
                    normalized_pos = (snr50 - acceptable_range[0]) / span if span > 0 else 0.5
                    normalized_pos = max(0.0, min(1.0, normalized_pos))
                    color_val = 30 + int(30 * (1 - normalized_pos))
                    curve_color = f'hsl({color_val}, 80%, 50%)'
                elif validity == 'Warning':
                    if snr50 > acceptable_range[1]:
                        normalized_pos = min((snr50 - acceptable_range[1]) / 5.0, 1.0)
                    else:
                        normalized_pos = min((acceptable_range[0] - snr50) / 5.0, 1.0)
                    color_val = int(30 * normalized_pos)
                    curve_color = f'hsl({color_val}, 90%, 50%)'
                elif validity == 'Fail':
                    curve_color = 'rgba(180, 180, 180, 0.4)'
                else:
                    curve_color = 'rgba(128, 128, 128, 0.5)'

                coef = (slope_pct_per_db / 100.0) * 4.0
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

                color_map = {
                    'Ideal': 'green',
                    'Acceptable': 'orange',
                    'Warning': 'red',
                    'Marginal': 'goldenrod',
                    'Fail': 'gray',
                    'Extrapolated': 'grey'
                }
                marker_color = color_map.get(validity, 'purple')

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

    title = "전체 문장들에 대한 Psychometric Function"

    fig.update_layout(
        title=title,
        xaxis_title="SNR Level (dB)",
        yaxis_title="정답률 (Correct Rate)",
        yaxis_range=[-0.05, 1.05],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest'
    )

    fig.update_traces(line=dict(width=1.5), selector=dict(mode='lines'))

    return fig
