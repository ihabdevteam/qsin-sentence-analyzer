"""실험 3(리스트 등가성) 전용 데이터 로더 및 분석 유틸리티.

기존 analysis_utils와 분리한 이유:
  1) analysis_utils의 로더는 세션 단위 SNR(test_reports_qsin.snr_level)을 읽지만,
     실험 3은 이 값이 항상 0이고 실제 SNR은 문장 단위(score_qsin.snr_level)에 있다.
  2) 실험 3 설계에서는 문장 1개가 SNR 1개에만 배치되므로 문장별 로지스틱 회귀가 불가능하다.
     역치는 Spearman-Karber 감산식으로 (참가자 x 리스트) 셀마다 직접 계산한다.
  3) 분석 단위가 문장이 아니라 리스트이며, 참가자가 반복측정 요인이다.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# --- 실험 3 사다리 설계 (assets/resources/exp3_lists.json과 일치) ---
LADDER = [-11.0, -8.0, -5.0, -2.0, 1.0, 4.0]
LADDER_MAX = 4.0            # 사다리 최고 SNR (dB)
LADDER_STEP = 3.0           # 사다리 간격 (dB)
KEYWORDS_PER_SENTENCE = 5   # 문장당 핵심어 수
SENTENCES_PER_LIST = 6
NUM_LISTS = 18
MAX_LIST_SCORE = SENTENCES_PER_LIST * KEYWORDS_PER_SENTENCE  # 30점

# --- 선별 기준 (로드맵 '선별 기준(Killion 차용)') ---
CRIT_SUBJECT_SD = 1.5       # 실험3: 리스트별 across-subject SD 상한 (dB)
CRIT_MEAN_DEV = 1.5         # 실험3: 리스트 평균이 전체 평균에서 벗어날 수 있는 범위 (dB)
CRIT_BETWEEN_LIST_SD = 1.0  # 실험4: 리스트 간 SNR-50 SD 상한 (dB)
CRIT_SCORE_SD = 1.5         # 실험4: 리스트 간 점수 SD 상한 (점)
CRIT_ANOVA_P = 0.05         # 실험4: ANOVA List 주효과는 유의하지 않아야 함

# K-Quick-SIN 정상 규준 (analysis_utils와 동일 값)
NORM_SNR50 = -4.16

TARGET_PARTICIPANTS = 30    # 로드맵 실험3 목표 인원 (25~30명)


# ============================================================
# 역치 환산
# ============================================================
def snr50_from_total(total_correct):
    """리스트 정답 핵심어 수(0~30)를 SNR-50(dB)으로 환산한다.

    Spearman-Karber 감산식: 최고SNR + 간격/2 - 간격 x (정답수 / 문장당 핵심어수)
      = 4 + 1.5 - 3 x (정답수 / 5) = 5.5 - 0.6 x 정답수

    표준 QuickSIN의 `25.5 - 정답수`는 사다리가 25dB에서 5dB 간격일 때의 식이므로
    실험 3(최고 +4dB, 3dB 간격)에는 그대로 쓸 수 없다.
    """
    step_per_keyword = LADDER_STEP / KEYWORDS_PER_SENTENCE
    return LADDER_MAX + LADDER_STEP / 2 - step_per_keyword * np.asarray(total_correct, dtype=float)


# ============================================================
# 데이터 로딩
# ============================================================
@st.cache_data(ttl=600)
def get_exp3_data(_supabase_client, experiment: str = 'exp3'):
    """실험 3 문장 단위 데이터를 조회하여 long 형식 DataFrame으로 반환한다.

    한 행 = 참가자 1명이 리스트 1개 안에서 들은 문장 1개.
    """
    page_size = 1000
    base_select = (
        "index, sentences, full_sentence, score, total_score, snr_level, trial_index,"
        " test_reports_qsin!inner(session_id, participant_no, sound_set,"
        " presentation_order, test_datetime, experiment)"
    )

    try:
        rows = []
        page = 0
        while True:
            start = page * page_size
            resp = (
                _supabase_client.table('score_qsin')
                .select(base_select)
                .eq('test_reports_qsin.experiment', experiment)
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1

        if not rows:
            return pd.DataFrame()

        records = []
        for item in rows:
            rep = item.get('test_reports_qsin') or {}
            records.append({
                'participant_no': rep.get('participant_no'),
                'session_id': rep.get('session_id'),
                'test_datetime': rep.get('test_datetime'),
                'list_no': rep.get('sound_set'),            # sound_set을 리스트 번호로 재활용
                'presentation_order': rep.get('presentation_order'),
                'sentence_id': item.get('index'),
                'trial_index': item.get('trial_index'),
                'snr_level': item.get('snr_level'),
                'total_score': item.get('total_score'),
                'score': item.get('score'),
                'sentences': item.get('sentences'),
                'full_sentence': item.get('full_sentence'),
            })

        df = pd.DataFrame(records)
        for col in ['participant_no', 'list_no', 'presentation_order', 'trial_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        for col in ['snr_level', 'total_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['num_keywords'] = df['sentences'].apply(lambda v: len(v) if isinstance(v, list) else 0)
        df['correct_rate'] = np.where(df['num_keywords'] > 0,
                                      df['total_score'] / df['num_keywords'], np.nan)
        return df.sort_values(['participant_no', 'presentation_order', 'trial_index'],
                              ignore_index=True)

    except Exception as e:
        st.error(f"실험 3 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


def build_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """문장 단위 데이터를 (참가자 x 리스트) 셀 단위로 집계한다.

    감산식은 사다리 6단계를 모두 제시했다는 전제에서만 성립한다. 중도 포기 등으로
    문장이 빠진 셀의 점수를 그대로 환산하면 빠진 문장을 오답으로 취급하는 셈이 되어
    역치가 크게 왜곡된다(쉬운 문장 3개가 빠지면 +7.8 dB까지 치우침). 따라서 불완전한
    셀은 snr50을 NaN으로 두고, is_complete로 구분한다.
    """
    if df.empty:
        return pd.DataFrame()

    cells = (
        df.dropna(subset=['participant_no', 'list_no'])
        .groupby(['participant_no', 'list_no'], as_index=False)
        .agg(total_correct=('total_score', 'sum'),
             sentences=('sentence_id', 'nunique'),
             snr_levels=('snr_level', 'nunique'),
             presentation_order=('presentation_order', 'first'),
             session_id=('session_id', 'first'))
    )
    cells['is_complete'] = (
        (cells['sentences'] == SENTENCES_PER_LIST) & (cells['snr_levels'] == len(LADDER))
    )
    cells['snr50'] = np.where(cells['is_complete'],
                              snr50_from_total(cells['total_correct']), np.nan)
    return cells


def cell_matrix(cells: pd.DataFrame, value: str = 'snr50') -> pd.DataFrame:
    """셀 테이블을 참가자(행) x 리스트(열) 행렬로 피벗한다."""
    if cells.empty:
        return pd.DataFrame()
    return cells.pivot(index='participant_no', columns='list_no', values=value).sort_index()


def complete_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """결측 셀이 없는 참가자만 남긴다(반복측정 분석용)."""
    if matrix.empty:
        return matrix
    return matrix.dropna(axis=0, how='any')


# ============================================================
# 무결성 검증
# ============================================================
def check_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """수집 데이터의 구조적 무결성을 점검하고 항목별 결과를 반환한다."""
    if df.empty:
        return pd.DataFrame()

    n_part = df['participant_no'].nunique()
    expected_rows = n_part * NUM_LISTS * SENTENCES_PER_LIST

    per_cell = df.groupby(['participant_no', 'list_no']).size()
    per_part_lists = df.groupby('participant_no')['list_no'].nunique()

    score_sum_mismatch = df.apply(
        lambda r: isinstance(r['score'], list)
        and abs(float(np.nansum(r['score'])) - float(r['total_score'])) > 1e-9,
        axis=1
    ).sum()

    ladder = sorted(LADDER)
    per_cell_ladder_ok = (
        df.groupby(['participant_no', 'list_no'])['snr_level']
        .apply(lambda s: sorted(pd.to_numeric(s, errors='coerce').dropna().tolist()) == ladder)
    )

    # (제시위치 x SNR) 조합 균형: 참가자 1명 기준 각 조합이 NUM_LISTS/6회씩 나와야 한다
    combo = df.groupby(['participant_no', 'trial_index', 'snr_level']).size()
    expected_combo = NUM_LISTS // len(LADDER)

    checks = [
        ("참가자 수", f"{n_part}명", None),
        ("전체 행 수", f"{len(df):,}행 / 기대 {expected_rows:,}행", len(df) == expected_rows),
        ("참가자당 리스트 18개", f"미달 {int((per_part_lists != NUM_LISTS).sum())}명",
         bool((per_part_lists == NUM_LISTS).all())),
        ("셀당 문장 6개", f"어긋난 셀 {int((per_cell != SENTENCES_PER_LIST).sum())}개",
         bool((per_cell == SENTENCES_PER_LIST).all())),
        ("문장별 SNR 결측", f"{int(df['snr_level'].isna().sum())}건",
         bool(df['snr_level'].isna().sum() == 0)),
        ("제시순서(trial_index) 결측", f"{int(df['trial_index'].isna().sum())}건",
         bool(df['trial_index'].isna().sum() == 0)),
        ("핵심어 5개", f"어긋난 행 {int((df['num_keywords'] != KEYWORDS_PER_SENTENCE).sum())}건",
         bool((df['num_keywords'] == KEYWORDS_PER_SENTENCE).all())),
        ("점수 합 = total_score", f"불일치 {int(score_sum_mismatch)}건", score_sum_mismatch == 0),
        ("리스트별 사다리 6단계", f"어긋난 셀 {int((~per_cell_ladder_ok).sum())}개",
         bool(per_cell_ladder_ok.all())),
        ("(제시위치 x SNR) 균형", f"기대 {expected_combo}회 / 어긋남 {int((combo != expected_combo).sum())}건",
         bool((combo == expected_combo).all())),
        ("중복 셀", f"{int(per_cell[per_cell > SENTENCES_PER_LIST].count())}개",
         bool((per_cell <= SENTENCES_PER_LIST).all())),
    ]
    return pd.DataFrame(checks, columns=['검증 항목', '결과', '통과'])


# ============================================================
# 선별 기준 판정
# ============================================================
def evaluate_lists(matrix: pd.DataFrame,
                   subject_sd_limit: float = CRIT_SUBJECT_SD,
                   mean_dev_limit: float = CRIT_MEAN_DEV) -> pd.DataFrame:
    """리스트별 SNR-50 통계와 선별 기준 통과 여부를 계산한다.

    기준 A: across-subject SD < subject_sd_limit (dB)
    기준 B: |리스트 평균 - 전체 평균| <= mean_dev_limit (dB)
    """
    if matrix.empty:
        return pd.DataFrame()

    n = matrix.notna().sum()
    mean = matrix.mean()
    sd = matrix.std(ddof=1)
    grand = float(np.nanmean(matrix.values))
    sem = sd / np.sqrt(n.where(n > 0))

    # SD의 95% 신뢰구간 (카이제곱) — 표본이 작을 때 판정이 이르다는 점을 함께 보여준다
    dfree = (n - 1).clip(lower=1)
    sd_lo = sd * np.sqrt(dfree / stats.chi2.ppf(0.975, dfree))
    sd_hi = sd * np.sqrt(dfree / stats.chi2.ppf(0.025, dfree))

    out = pd.DataFrame({
        'list_no': matrix.columns,
        'n': n.values,
        'snr50_mean': mean.values,
        'snr50_sd': sd.values,
        'snr50_sem': sem.values,
        'sd_ci_low': sd_lo.values,
        'sd_ci_high': sd_hi.values,
        'dev_from_grand': (mean - grand).values,
        'snr50_min': matrix.min().values,
        'snr50_max': matrix.max().values,
    })
    # 평균 정답 핵심어 수로 환산 (표시용)
    out['mean_score'] = (LADDER_MAX + LADDER_STEP / 2 - out['snr50_mean']) \
        / (LADDER_STEP / KEYWORDS_PER_SENTENCE)

    out['pass_sd'] = out['snr50_sd'] < subject_sd_limit
    out['pass_dev'] = out['dev_from_grand'].abs() <= mean_dev_limit
    out['pass_all'] = out['pass_sd'] & out['pass_dev']
    # SD 신뢰구간이 기준선을 걸치면 판정 보류 대상
    out['sd_ci_spans_limit'] = (out['sd_ci_low'] < subject_sd_limit) & (out['sd_ci_high'] >= subject_sd_limit)
    return out.sort_values('snr50_mean', ignore_index=True)


def summary_criteria(matrix: pd.DataFrame, anova: dict | None = None) -> pd.DataFrame:
    """전체 수준 선별 기준(리스트 간 SD, 점수 SD, ANOVA)의 충족 여부를 정리한다."""
    if matrix.empty:
        return pd.DataFrame()

    list_means = matrix.mean()
    between_sd = float(list_means.std(ddof=1))
    score_means = (LADDER_MAX + LADDER_STEP / 2 - list_means) / (LADDER_STEP / KEYWORDS_PER_SENTENCE)
    score_sd = float(score_means.std(ddof=1))
    grand = float(np.nanmean(matrix.values))

    rows = [
        ("전체 평균 SNR-50", f"{grand:+.2f} dB", f"규준 {NORM_SNR50:+.2f} dB", None),
        ("리스트 간 SNR-50 SD", f"{between_sd:.2f} dB", f"< {CRIT_BETWEEN_LIST_SD:.1f} dB (실험4)",
         between_sd < CRIT_BETWEEN_LIST_SD),
        ("리스트 간 점수 SD", f"{score_sd:.2f} 점", f"< {CRIT_SCORE_SD:.1f} 점 (실험4)",
         score_sd < CRIT_SCORE_SD),
    ]
    if anova and anova.get('p') is not None:
        rows.append(("ANOVA List 주효과 (무보정)", f"p = {anova['p']:.4f}",
                     f"> {CRIT_ANOVA_P}", anova['p'] > CRIT_ANOVA_P))
        if anova.get('p_gg') is not None:
            rows.append(("ANOVA List 주효과 (GG 보정)", f"p = {anova['p_gg']:.4f}",
                         f"> {CRIT_ANOVA_P}", anova['p_gg'] > CRIT_ANOVA_P))
    return pd.DataFrame(rows, columns=['지표', '현재 값', '기준', '충족'])


# ============================================================
# 반복측정 ANOVA / 사후검정
# ============================================================
def repeated_measures_anova(matrix: pd.DataFrame) -> dict:
    """리스트를 피험자 내 요인으로 하는 반복측정 ANOVA.

    결측 셀이 있는 참가자는 제외한다. Greenhouse-Geisser 보정을 함께 반환하되,
    참가자 수가 리스트 수보다 적으면 구형성 추정 자체가 불안정하다(unstable 플래그).
    """
    mat = complete_matrix(matrix)
    n, k = mat.shape
    if n < 2 or k < 2:
        return {'status': 'Error: Not Enough Data', 'n': n, 'k': k}

    Y = mat.values.astype(float)
    grand = Y.mean()
    row_m = Y.mean(axis=1, keepdims=True)
    col_m = Y.mean(axis=0, keepdims=True)

    ss_list = n * ((col_m - grand) ** 2).sum()
    ss_subj = k * ((row_m - grand) ** 2).sum()
    ss_err = ((Y - row_m - col_m + grand) ** 2).sum()
    df_list, df_err, df_subj = k - 1, (k - 1) * (n - 1), n - 1

    ms_list, ms_err = ss_list / df_list, ss_err / df_err
    f_val = ms_list / ms_err if ms_err > 0 else np.nan
    p_val = float(stats.f.sf(f_val, df_list, df_err)) if np.isfinite(f_val) else None

    # Greenhouse-Geisser epsilon
    cov = np.cov(Y.T, ddof=1)
    dev = cov - cov.mean(axis=0, keepdims=True) - cov.mean(axis=1, keepdims=True) + cov.mean()
    denom = (k - 1) * (dev ** 2).sum()
    eps = float(np.trace(dev) ** 2 / denom) if denom > 0 else 1.0
    eps = min(max(eps, 1.0 / (k - 1)), 1.0)
    p_gg = float(stats.f.sf(f_val, df_list * eps, df_err * eps)) if np.isfinite(f_val) else None

    # Fisher LSD (dB) 및 리스트 평균의 95% CI 반폭
    t_crit = float(stats.t.ppf(0.975, df_err))
    lsd = t_crit * np.sqrt(2 * ms_err / n)
    ci_half = t_crit * np.sqrt(ms_err / n)

    return {
        'status': 'Success',
        'n': n, 'k': k,
        'F': float(f_val), 'df_list': df_list, 'df_err': df_err,
        'p': p_val, 'eps_gg': eps, 'p_gg': p_gg,
        'eta2_p': float(ss_list / (ss_list + ss_err)) if (ss_list + ss_err) > 0 else None,
        'ms_err': float(ms_err),
        'lsd': float(lsd), 'ci_half': float(ci_half),
        'icc_single': float((ss_subj / df_subj - ms_err) / (ss_subj / df_subj + (k - 1) * ms_err))
        if (ss_subj / df_subj + (k - 1) * ms_err) > 0 else None,
        'sphericity_unstable': n <= k,
    }


def posthoc_vs_grand(list_stats: pd.DataFrame, anova: dict) -> pd.DataFrame:
    """각 리스트 평균이 전체 평균과 유의하게 다른지 LSD 기준으로 표시한다."""
    if list_stats.empty or anova.get('status') != 'Success':
        return list_stats
    out = list_stats.copy()
    threshold = anova['lsd'] / np.sqrt(2)  # 전체 평균과의 비교이므로 LSD/sqrt(2)
    out['deviates'] = out['dev_from_grand'].abs() > threshold
    out['lsd_threshold'] = threshold
    return out


# ============================================================
# 순서(연습) 효과
# ============================================================
def order_effect(cells: pd.DataFrame) -> dict:
    """제시 위치와 성적의 관계를 추정한다.

    유의성 판단은 참가자별 회귀 기울기를 구해 1표본 t검정하는 방식(참가자 수준)을
    쓴다. 셀을 독립 관측으로 보는 상관계수 p값은 한 참가자가 18셀을 만드는 구조를
    무시해 불확실성을 과소평가한다(실측: p=.001 → 참가자 수준 p=.080). 상관계수는
    효과 크기를 보여주는 기술 통계로만 남긴다.
    """
    df = cells.dropna(subset=['presentation_order', 'snr50', 'participant_no']).copy()
    if df.empty or df['participant_no'].nunique() < 2:
        return {'status': 'Error: Not Enough Data'}

    df['snr50_c'] = df['snr50'] - df.groupby('participant_no')['snr50'].transform('mean')
    df['order_c'] = (df['presentation_order'].astype(float)
                     - df.groupby('participant_no')['presentation_order'].transform('mean').astype(float))
    if df['order_c'].std(ddof=1) == 0:
        return {'status': 'Error: No Order Variation'}

    r, p_pooled = stats.pearsonr(df['order_c'], df['snr50_c'])
    slope = float(np.polyfit(df['order_c'], df['snr50_c'], 1)[0])

    # 참가자별 기울기 → 1표본 t검정 (참가자를 분석 단위로)
    slopes = []
    for _, g in df.groupby('participant_no'):
        if g['presentation_order'].nunique() > 2:
            slopes.append(float(np.polyfit(g['presentation_order'].astype(float), g['snr50'], 1)[0]))
    slopes = np.asarray(slopes, dtype=float)
    if len(slopes) >= 2:
        t_stat, p_clustered = stats.ttest_1samp(slopes, 0.0)
        t_stat, p_clustered = float(t_stat), float(p_clustered)
    else:
        t_stat, p_clustered = None, None

    # 카운터밸런싱 균형: 리스트별 평균 제시 위치 (완전 균형이면 모두 (NUM_LISTS+1)/2)
    balance = (df.groupby('list_no')['presentation_order'].mean()
               .rename('mean_position').reset_index())
    ideal = (NUM_LISTS + 1) / 2

    return {
        'status': 'Success',
        'r': float(r),
        'p_pooled': float(p_pooled),          # 기술 통계용. 독립성 가정이 깨져 과소평가됨
        'p': p_clustered,                     # 판단 기준: 참가자 수준 검정
        't': t_stat,
        'n_participants': int(len(slopes)),
        'n_negative': int((slopes < 0).sum()),
        'n_positive': int((slopes > 0).sum()),
        'slope_db_per_position': slope,
        'total_db_over_lists': slope * (NUM_LISTS - 1),
        'balance': balance,
        'ideal_position': ideal,
        'balance_spread': float(balance['mean_position'].max() - balance['mean_position'].min()),
        'n_cells': int(len(df)),
    }


# ============================================================
# 문장 단위 점검 (3-3 문장 제거용)
# ============================================================
def sentence_stats(df: pd.DataFrame, flag_threshold: float = 0.20) -> pd.DataFrame:
    """문장별 정답률을 같은 SNR의 전체 평균과 비교한다.

    실험 3에서 문장은 SNR 1개에만 배치되므로 문장별 SNR-50은 추정할 수 없다.
    대신 '같은 SNR에 놓인 다른 문장들과 비교해 유독 쉽거나 어려운가'를 본다.
    """
    if df.empty:
        return pd.DataFrame()

    snr_mean = df.groupby('snr_level')['correct_rate'].mean().rename('snr_mean_rate')
    out = (df.groupby(['sentence_id', 'snr_level', 'list_no'], as_index=False)
             .agg(n=('correct_rate', 'size'),
                  rate=('correct_rate', 'mean'),
                  rate_sd=('correct_rate', lambda s: s.std(ddof=1) if s.count() > 1 else 0.0),
                  full_sentence=('full_sentence', 'first')))
    out = out.merge(snr_mean, on='snr_level', how='left')
    out['dev_from_snr_mean'] = out['rate'] - out['snr_mean_rate']
    out['flag'] = np.where(out['dev_from_snr_mean'].abs() >= flag_threshold, '편차 큼', '')
    return out.sort_values('dev_from_snr_mean', ignore_index=True)


# ============================================================
# 신뢰도
# ============================================================
def reliability(matrix: pd.DataFrame) -> dict:
    """리스트를 문항으로 보는 내적 일관성과, 리스트 평균 개수별 측정 정밀도를 계산한다.

    정밀도에는 참가자·리스트 주효과를 제거한 오차항 sqrt(MS_error)을 쓴다.
    리스트별 피험자 간 SD(within_sd)에는 '사람마다 실력이 다른 몫'이 섞여 있어
    개인 역치의 측정 정밀도를 과대평가한다.
    """
    mat = complete_matrix(matrix)
    n, k = mat.shape
    if n < 3 or k < 2:
        return {'status': 'Error: Not Enough Data'}

    Y = mat.values.astype(float)
    var_total = Y.sum(axis=1).var(ddof=1)
    var_items = Y.var(axis=0, ddof=1).sum()
    alpha = (k / (k - 1)) * (1 - var_items / var_total) if var_total > 0 else None

    grand = Y.mean()
    row_m = Y.mean(axis=1, keepdims=True)
    col_m = Y.mean(axis=0, keepdims=True)
    ms_err = ((Y - row_m - col_m + grand) ** 2).sum() / ((k - 1) * (n - 1))
    error_sd = float(np.sqrt(ms_err))                           # 순수 측정오차
    within_sd = float(np.sqrt(np.mean(Y.var(axis=0, ddof=1))))  # 참조용(실력차 포함)
    precision = {m: 1.96 * error_sd / np.sqrt(m) for m in (1, 2, 3, 4)}

    return {
        'status': 'Success',
        'n': n, 'k': k,
        'alpha': float(alpha) if alpha is not None else None,
        'error_sd': error_sd,
        'within_sd': within_sd,
        'subject_sd': float(Y.mean(axis=1).std(ddof=1)),
        'precision_95ci': precision,
    }
