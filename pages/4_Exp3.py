"""실험 3 — 리스트 등가성 검증 및 선별 기준 판정 페이지."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.db_utils import init_supabase_client
from modules.exp3_utils import (
    CRIT_ANOVA_P,
    CRIT_MEAN_DEV,
    CRIT_SUBJECT_SD,
    EXCLUDED_SESSIONS,
    LADDER,
    NORM_SNR50,
    NUM_LISTS,
    TARGET_PARTICIPANTS,
    build_cell_table,
    cell_matrix,
    check_integrity,
    complete_matrix,
    evaluate_lists,
    get_exp3_data,
    order_effect,
    posthoc_vs_grand,
    reliability,
    repeated_measures_anova,
    sentence_stats,
    summary_criteria,
)

st.set_page_config(page_title="실험 3 리스트 등가성", layout="wide")
st.title("실험 3 — 리스트 등가성 검증 (Exp3)")
st.write(
    "참가자 번호별 Williams 순서로 수집된 18리스트 데이터를 리스트 단위로 분석하고, "
    "선별 기준(across-subject SD, 전체 평균 편차, 리스트 간 SD, ANOVA)의 충족 여부를 판정합니다."
)

ACCENT_OK = "#0c6265"
ACCENT_NG = "#9c3b2e"

supabase = init_supabase_client()
if not supabase:
    st.stop()

if 'exp3_df' not in st.session_state:
    st.session_state.exp3_df = None


def _pass_icon(v):
    """통과 여부를 기호로 변환한다."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "✅" if bool(v) else "❌"


# ============================================================
# 0. 데이터 조회 및 수집 현황
# ============================================================
st.header("0. 데이터 조회 및 수집 현황")

col_btn, col_chk, col_opt = st.columns([1, 1, 2])
with col_btn:
    fetch = st.button("실험 3 데이터 조회", type="primary")
with col_chk:
    fresh = st.checkbox("캐시 무시", value=False,
                        help="참가자를 방금 추가했다면 체크하세요. 조회 결과는 10분간 캐시됩니다.")
with col_opt:
    st.caption(
        "experiment='exp3' 태그가 달린 리포트만 조회합니다. "
        "문장별 SNR은 score_qsin.snr_level에서 읽습니다(리포트의 snr_level은 실험 3에서 항상 0)."
    )
if fetch:
    if fresh:
        get_exp3_data.clear()
    with st.spinner("Supabase에서 실험 3 데이터를 조회 중입니다..."):
        try:
            st.session_state.exp3_df = get_exp3_data(supabase)
        except Exception as e:
            st.error(f"실험 3 데이터를 불러오는 중 오류가 발생했습니다: {e}")
            st.stop()

df = st.session_state.exp3_df
if df is None:
    st.info("먼저 위의 **실험 3 데이터 조회** 버튼을 눌러주세요.")
    st.stop()
if df.empty:
    st.warning("조회된 실험 3 데이터가 없습니다.")
    st.stop()

cells_all = build_cell_table(df)
n_part = int(df['participant_no'].nunique())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("참가자", f"{n_part}명", help=(f"목표 {TARGET_PARTICIPANTS}명 = Williams 1주기. 제외자가 생기면 같은 행에 "
                      "교체자를 넣고 번호는 '원래 번호 + 18'을 부여합니다."))
c2.metric("목표 진행률", f"{n_part / TARGET_PARTICIPANTS * 100:.0f}%")
c3.metric("리포트(셀)", f"{len(cells_all):,}건")
c4.metric("문장 기록", f"{len(df):,}행")
c5.metric("수집 기간", f"{str(df['test_datetime'].min())[:10]} ~ {str(df['test_datetime'].max())[:10]}")

st.progress(min(n_part / TARGET_PARTICIPANTS, 1.0))

if EXCLUDED_SESSIONS:
    with st.expander(f"본분석 제외 세션 {len(EXCLUDED_SESSIONS)}개 (DB에는 보존)"):
        st.dataframe(
            pd.DataFrame(list(EXCLUDED_SESSIONS.items()), columns=['세션', '제외 사유']),
            hide_index=True, width='stretch'
        )

# Williams 균형 안내 — 18의 배수에서만 제시 순서가 완전 균형이 된다
if n_part % NUM_LISTS != 0:
    cycles = n_part / NUM_LISTS
    st.warning(
        f"**카운터밸런싱 미완성** — Williams {NUM_LISTS}x{NUM_LISTS} 설계는 참가자 수가 "
        f"{NUM_LISTS}의 배수일 때만 제시 순서가 완전히 균형을 이룹니다 (현재 {n_part}명 = {cycles:.2f}주기). "
        f"지금은 리스트 난이도와 순서(연습)효과가 부분적으로 교락되어 있습니다. §4에서 균형도를 확인하세요."
    )

with st.expander("참가자별 수집 내역"):
    part_tbl = (
        cells_all.groupby('participant_no', as_index=False)
        .agg(세션=('session_id', 'first'),
             리스트수=('list_no', 'nunique'),
             총정답=('total_correct', 'sum'),
             평균SNR50=('snr50', 'mean'))
    )
    part_tbl['평균SNR50'] = part_tbl['평균SNR50'].round(2)
    st.dataframe(part_tbl, hide_index=True, width='stretch')

# --- 분석 대상 참가자 선택 ---
all_parts = sorted(df['participant_no'].dropna().unique().tolist())
sel_parts = st.multiselect(
    "분석에 포함할 참가자", options=all_parts, default=all_parts,
    help="제외하면 아래 모든 분석이 선택된 참가자만으로 다시 계산됩니다."
)
if not sel_parts:
    st.warning("참가자를 1명 이상 선택해주세요.")
    st.stop()

df_sel = df[df['participant_no'].isin(sel_parts)]
cells = build_cell_table(df_sel)
matrix_full = cell_matrix(cells, 'snr50')

# 사다리 6단계를 다 제시하지 못한 셀은 감산식이 성립하지 않아 역치가 NaN이다
incomplete = cells[~cells['is_complete']]
if not incomplete.empty:
    st.error(
        f"**불완전한 셀 {len(incomplete)}개** — 정확히 6문장·사다리 {len(LADDER)}단계 각 1개가 "
        "아닌 셀(누락 또는 중복 저장)은 감산식이 성립하지 않아 SNR-50을 계산하지 않았습니다"
        "(빠진 문장을 오답으로 취급하면 역치가 최대 7.8 dB까지 왜곡됩니다). SNR-50을 쓰는 "
        "분석(§2 기준 판정·ANOVA, §3 시뮬레이터, §4 순서 효과, §5 신뢰도)에서 제외되며, "
        "문장 단위 집계(§1 무결성 검증, §3 문장 점검, 심리측정 함수, 내보내기)에는 포함됩니다."
    )
    st.dataframe(
        incomplete[['participant_no', 'list_no', 'rows', 'sentences', 'snr_levels',
                    'total_correct', 'presentation_order', 'session_id']],
        hide_index=True, width='stretch'
    )

# ============================================================
# 1. 데이터 무결성 검증
# ============================================================
st.header("1. 데이터 무결성 검증")
integrity = check_integrity(df_sel)
failed = integrity[integrity['통과'] == False]  # noqa: E712

if failed.empty:
    st.success(f"무결성 검증 {len(integrity) - 1}개 항목 모두 통과 — 분석 가능한 상태입니다.")
else:
    st.error(f"무결성 검증 {len(failed)}개 항목 실패 — 아래 표에서 확인하세요.")

integrity_view = integrity.copy()
integrity_view['통과'] = integrity_view['통과'].map(_pass_icon)
st.dataframe(integrity_view, hide_index=True, width='stretch')

# ============================================================
# 2. 선별 기준 판정
# ============================================================
st.header("2. 선별 기준 판정")

with st.expander("기준 설정 변경", expanded=False):
    cc1, cc2 = st.columns(2)
    sd_limit = cc1.number_input(
        "기준 A — across-subject SD 상한 (dB)", min_value=0.5, max_value=5.0,
        value=float(CRIT_SUBJECT_SD), step=0.1,
        help="리스트별 피험자 간 SNR-50 표준편차의 상한. 크면 측정이 불안정한 리스트입니다."
    )
    dev_limit = cc2.number_input(
        "기준 B — 전체 평균과의 허용 편차 (dB)", min_value=0.5, max_value=5.0,
        value=float(CRIT_MEAN_DEV), step=0.1,
        help="리스트 평균 SNR-50이 전체 평균에서 벗어날 수 있는 범위. 크면 난이도가 치우친 리스트입니다."
    )

anova = repeated_measures_anova(matrix_full)
list_stats = evaluate_lists(matrix_full, sd_limit, dev_limit)
if anova.get('status') == 'Success':
    list_stats = posthoc_vs_grand(list_stats, anova)

# --- 전체 수준 기준 ---
st.subheader("전체 수준 지표")
summary = summary_criteria(matrix_full, anova)
summary_view = summary.copy()
summary_view['충족'] = summary_view['충족'].map(_pass_icon)
st.dataframe(summary_view, hide_index=True, width='stretch')
st.caption(
    "ANOVA 기준(p > 0.05)은 로드맵에 명시된 기준을 그대로 적용한 것입니다. 다만 "
    "**차이를 검출하지 못한 것이 등가성의 증거는 아닙니다** — 표본이 작으면 실제로 "
    "리스트 간 차이가 있어도 이 기준을 통과합니다. 등가성을 주장하려면 허용 한계를 "
    "정한 등가성 검정(TOST 등)이 필요합니다."
)

if anova.get('status') == 'Success':
    n_total = matrix_full.shape[0]
    if anova['n'] < n_total:
        st.warning(
            f"**ANOVA 제외** — 18리스트를 모두 마치지 못한 참가자 {n_total - anova['n']}명은 "
            f"반복측정 분석에서 빠졌습니다(사용 {anova['n']}명 / 선택 {n_total}명). "
            "§1 무결성 검증에서 누락된 셀을 확인하세요. 리스트별 판정표의 n은 영향받지 않습니다."
        )
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ANOVA F", f"{anova['F']:.2f}",
              help=f"F({anova['df_list']}, {anova['df_err']})")
    a2.metric("p (무보정)", f"{anova['p']:.4f}",
              delta="기준 충족" if anova['p'] > CRIT_ANOVA_P else "기준 미달",
              delta_color="normal" if anova['p'] > CRIT_ANOVA_P else "inverse")
    a3.metric("p (GG 보정)", f"{anova['p_gg']:.4f}",
              help=f"Greenhouse-Geisser epsilon = {anova['eps_gg']:.3f}")
    a4.metric("부분 eta²", f"{anova['eta2_p']:.3f}")

    if anova.get('sphericity_unstable'):
        st.info(
            f"**구형성 추정 주의** — 참가자 {anova['n']}명 ≤ 리스트 {anova['k']}개이므로 "
            f"공분산 행렬 추정이 불안정하고, GG epsilon({anova['eps_gg']:.2f})이 과도하게 작게 나옵니다. "
            "무보정 p는 낙관적, GG p는 과도하게 보수적인 구간입니다. 참가자가 리스트 수를 넘어서면 해소됩니다."
        )
else:
    st.warning(f"반복측정 ANOVA를 계산할 수 없습니다: {anova.get('status')}")

# --- 리스트별 판정 ---
st.subheader("리스트별 판정")
n_pass = int(list_stats['pass_all'].sum())
p1, p2, p3 = st.columns(3)
p1.metric("기준 A·B 모두 통과", f"{n_pass} / {len(list_stats)}개")
p2.metric("기준 A(SD) 미달", f"{int((~list_stats['pass_sd']).sum())}개")
p3.metric("기준 B(편차) 미달", f"{int((~list_stats['pass_dev']).sum())}개")

n_span = int(list_stats['sd_ci_spans_limit'].sum())
if n_span:
    st.caption(
        f"⚠️ {n_span}개 리스트는 SD의 95% 신뢰구간이 기준선({sd_limit:.1f} dB)을 걸칩니다 — "
        "표본이 늘면 판정이 뒤집힐 수 있는 리스트입니다."
    )

view = list_stats.copy()
view['판정'] = view['pass_all'].map(lambda v: "통과" if v else "미달")
view['SD 95% CI'] = view.apply(lambda r: f"{r['sd_ci_low']:.2f} – {r['sd_ci_high']:.2f}", axis=1)
if 'deviates' in view.columns:
    view['평균과 차이(LSD·보정 전)'] = view['deviates'].map(lambda v: "유의" if v else "-")
else:
    view['평균과 차이(LSD·보정 전)'] = "-"
view = view[['list_no', 'n', 'mean_score', 'snr50_mean', 'dev_from_grand', 'snr50_sd',
             'SD 95% CI', 'snr50_min', 'snr50_max', 'pass_sd', 'pass_dev', '평균과 차이(LSD·보정 전)', '판정']]
view.columns = ['리스트', 'n', '평균 정답수', 'SNR-50(dB)', '전체평균 편차',
                'SD(dB)', 'SD 95% CI', '최소', '최대', '기준A', '기준B', '평균과 차이(LSD·보정 전)', '판정']
for c in ['평균 정답수', 'SNR-50(dB)', '전체평균 편차', 'SD(dB)', '최소', '최대']:
    view[c] = view[c].astype(float).round(2)
view['기준A'] = view['기준A'].map(_pass_icon)
view['기준B'] = view['기준B'].map(_pass_icon)

st.dataframe(
    view.style.apply(
        lambda row: ['background-color: rgba(156,59,46,0.10)' if row['판정'] == '미달' else '' for _ in row],
        axis=1
    ),
    hide_index=True, width='stretch'
)

# --- forest plot ---
st.subheader("리스트별 SNR-50 (평균 ± 95% CI)")
fp = list_stats.sort_values('snr50_mean')
grand_mean = float(np.nanmean(matrix_full.values))

# ANOVA의 통합 오차항은 완전 케이스만으로 계산되므로, 결측 셀이 있으면 각 리스트의
# 평균과 신뢰구간이 서로 다른 표본을 가리키게 된다. 그럴 때는 리스트별 SEM을 쓴다.
all_complete = anova.get('status') == 'Success' and anova['n'] == matrix_full.shape[0]
if all_complete:
    ci = np.full(len(fp), anova['ci_half'])
    ci_note = "통합 오차항(MS_error) 기준"
else:
    ci = fp['snr50_ci_half'].to_numpy()
    ci_note = "리스트별 SEM 기준 (결측 셀이 있어 통합 오차항을 쓰지 않음)"

fig = go.Figure()
fig.add_vrect(x0=grand_mean - dev_limit, x1=grand_mean + dev_limit,
              fillcolor="rgba(12,98,101,0.08)", line_width=0,
              annotation_text=f"기준 B 허용 범위 (±{dev_limit:.1f} dB)",
              annotation_position="top left")
fig.add_vline(x=grand_mean, line_dash="solid", line_color=ACCENT_OK,
              annotation_text=f"전체 평균 {grand_mean:.2f}", annotation_position="bottom right")
fig.add_vline(x=NORM_SNR50, line_dash="dot", line_color="#8b9b9e",
              annotation_text=f"K-QuickSIN 규준 {NORM_SNR50:.2f}", annotation_position="top right")
fig.add_trace(go.Scatter(
    x=fp['snr50_mean'], y=[f"리스트 {int(v)}" for v in fp['list_no']],
    error_x=dict(type='data', array=ci, color="#5d7076", thickness=1.2, width=4),
    mode='markers',
    marker=dict(size=10,
                color=[ACCENT_OK if ok else ACCENT_NG for ok in fp['pass_all']],
                line=dict(width=0)),
    customdata=np.stack([fp['snr50_sd'], fp['mean_score'], fp['n']], axis=-1),
    hovertemplate=("%{y}<br>SNR-50 %{x:.2f} dB<br>SD %{customdata[0]:.2f} dB"
                   "<br>평균 정답수 %{customdata[1]:.1f}<br>n=%{customdata[2]}<extra></extra>"),
    name="리스트 평균"
))
fig.update_layout(height=max(380, 24 * len(fp)), margin=dict(l=10, r=10, t=30, b=10),
                  xaxis_title="SNR-50 (dB) — 왼쪽이 쉬운 리스트", yaxis_title=None,
                  showlegend=False)
st.plotly_chart(fig)
st.caption(f"오차 막대는 95% 신뢰구간 — {ci_note}.")

# --- 히트맵 ---
st.subheader("참가자 × 리스트 SNR-50 히트맵")
hm = matrix_full.copy()
fig_hm = go.Figure(go.Heatmap(
    z=hm.values,
    x=[f"L{int(c)}" for c in hm.columns],
    y=[f"P{int(i)}" for i in hm.index],
    colorscale="RdBu", zmid=grand_mean, reversescale=True,
    colorbar=dict(title="SNR-50<br>(dB)"),
    hovertemplate="%{y} · %{x}<br>SNR-50 %{z:.2f} dB<extra></extra>"
))
fig_hm.update_layout(height=max(300, 30 * len(hm) + 120), margin=dict(l=10, r=10, t=20, b=10),
                     xaxis_title="리스트", yaxis_title="참가자")
st.plotly_chart(fig_hm)

# ============================================================
# 3. 재구성 시뮬레이터 (실험 3-3)
# ============================================================
st.header("3. 재구성 시뮬레이터 (실험 3-3)")
st.write("등가성에서 벗어난 리스트·문장을 제외했을 때 선별 기준이 어떻게 변하는지 즉시 재계산합니다.")

if 'exp3_excl_widget' not in st.session_state:
    st.session_state.exp3_excl_widget = []

b1, b2 = st.columns([1, 1])
# 위젯 생성 전에 세션 상태를 직접 바꿔야 multiselect에 반영된다
if b1.button("기준 미달 리스트 자동 선택"):
    st.session_state.exp3_excl_widget = [
        int(v) for v in list_stats.loc[~list_stats['pass_all'], 'list_no']
    ]
if b2.button("선택 초기화"):
    st.session_state.exp3_excl_widget = []

excluded = st.multiselect(
    "제외할 리스트", options=[int(c) for c in matrix_full.columns],
    key='exp3_excl_widget'
)

kept = [c for c in matrix_full.columns if int(c) not in excluded]
if len(kept) < 3:
    st.warning("리스트를 3개 이상 남겨주세요.")
else:
    mat_k = matrix_full[kept]
    anova_k = repeated_measures_anova(mat_k)
    stats_k = evaluate_lists(mat_k, sd_limit, dev_limit)
    sum_k = summary_criteria(mat_k, anova_k)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("남은 리스트", f"{len(kept)}개", delta=f"-{len(excluded)}개" if excluded else None)
    between_before = float(matrix_full.mean().std(ddof=1))
    between_after = float(mat_k.mean().std(ddof=1))
    s2.metric("리스트 간 SD", f"{between_after:.2f} dB",
              delta=f"{between_after - between_before:+.2f}", delta_color="inverse")
    sd_before, sd_after = list_stats['snr50_sd'].max(), stats_k['snr50_sd'].max()
    s3.metric("최대 across-subj SD", f"{sd_after:.2f} dB",
              delta=f"{sd_after - sd_before:+.2f}", delta_color="inverse")
    if anova_k.get('status') == 'Success' and anova.get('status') == 'Success':
        s4.metric("ANOVA p (무보정)", f"{anova_k['p']:.4f}",
                  delta=f"{anova_k['p'] - anova['p']:+.4f}")

    sum_k_view = sum_k.copy()
    sum_k_view['충족'] = sum_k_view['충족'].map(_pass_icon)
    st.dataframe(sum_k_view, hide_index=True, width='stretch')

    st.caption(
        "SD가 큰 리스트를 제외하면 오차항(MS_error)이 함께 줄어들어 ANOVA의 F가 커지고 "
        "p가 오히려 작아질 수 있습니다. 기준 A와 ANOVA 기준은 같은 방향으로 움직이지 않으므로 "
        "두 지표를 함께 보고 판단하세요."
    )

    with st.expander("문장 단위 점검 (문장 제거 후보)"):
        st.caption(
            "실험 3에서 문장은 SNR 1개에만 배치되므로 문장별 SNR-50은 추정할 수 없습니다. "
            "대신 같은 SNR에 놓인 문장들의 평균 정답률과 비교해 유독 쉽거나 어려운 문장을 찾습니다."
        )
        thr = st.slider("편차 임계값 (%p)", min_value=5, max_value=40, value=20, step=5,
                        help="같은 SNR의 평균 정답률과 이만큼 차이 나는 문장을 표시합니다.") / 100
        sent = sentence_stats(df_sel[~df_sel['list_no'].isin(excluded)], flag_threshold=thr)
        only_flagged = st.checkbox(f"편차 큰 문장만 보기 (±{thr * 100:.0f}%p 이상)", value=True)
        st.caption(f"임계값 {thr * 100:.0f}%p 기준 {int((sent['flag'] != '').sum())}건 해당")
        sv = sent[sent['flag'] != ''] if only_flagged else sent
        sv = sv[['sentence_id', 'list_no', 'snr_level', 'n', 'rate', 'snr_mean_rate',
                 'dev_from_snr_mean', 'full_sentence']].copy()
        sv.columns = ['문장ID', '리스트', 'SNR(dB)', 'n', '정답률', '동일SNR 평균', '편차', '문장']
        for c in ['정답률', '동일SNR 평균', '편차']:
            sv[c] = (sv[c].astype(float) * 100).round(1)
        st.dataframe(sv, hide_index=True, width='stretch')

# ============================================================
# 4. 순서(연습) 효과 및 카운터밸런싱
# ============================================================
st.header("4. 순서(연습) 효과 및 카운터밸런싱")
oe = order_effect(cells)

if oe.get('status') != 'Success':
    st.warning(f"순서 효과를 계산할 수 없습니다: {oe.get('status')}")
else:
    sig = oe['p'] is not None and oe['p'] < 0.05
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("기울기", f"{oe['slope_db_per_position']:+.3f} dB/위치",
              help="참가자별 회귀 기울기의 평균. 음수 = 뒤로 갈수록 역치가 낮아짐(개선)")
    o2.metric("18리스트 누적", f"{oe['total_db_over_lists']:+.2f} dB")
    o3.metric("p (참가자 수준)", "–" if oe['p'] is None else f"{oe['p']:.4f}",
              delta=("유의" if sig else "비유의") if oe['p'] is not None else None,
              delta_color="inverse" if sig else "normal",
              help=(f"참가자별 기울기 {oe['n_participants']}개의 1표본 t검정"
                    + (f" (t={oe['t']:.2f})" if oe['t'] is not None else "")))
    o4.metric("기울기 부호", f"음 {oe['n_negative']} / 양 {oe['n_positive']}",
              help=f"상관 r={oe['r']:+.3f} (셀 단위, 효과 크기 참고용)")

    if sig:
        st.warning(
            f"**순서 효과 유의** — 검사를 진행할수록 역치가 "
            f"{abs(oe['total_db_over_lists']):.2f} dB 변합니다(연습 효과). "
            "Williams 설계가 이를 상쇄하려면 참가자 수가 리스트 수의 배수여야 합니다."
        )
    elif oe['p'] is not None:
        st.info(
            f"**순서 효과 비유의** — 기울기는 {oe['slope_db_per_position']:+.3f} dB/위치"
            f"(18리스트 누적 {oe['total_db_over_lists']:+.2f} dB)로 경향은 보이지만, "
            f"참가자 수준 검정에서는 유의하지 않습니다(p={oe['p']:.3f}, "
            f"{oe['n_participants']}명 중 음수 기울기 {oe['n_negative']}명). "
            "참가자가 늘면 판단이 달라질 수 있습니다."
        )
    st.caption(
        f"셀 단위 상관의 p값(p={oe['p_pooled']:.4f})은 한 참가자가 18셀을 만드는 구조를 "
        "무시해 불확실성을 과소평가합니다. 판단은 참가자 수준 검정을 기준으로 하세요."
    )

    bal = oe['balance'].copy()
    bal['편차'] = bal['mean_position'] - oe['ideal_position']
    fig_b = go.Figure()
    fig_b.add_hline(y=oe['ideal_position'], line_dash="dash", line_color=ACCENT_OK,
                    annotation_text=f"완전 균형 = {oe['ideal_position']:.1f}")
    fig_b.add_trace(go.Bar(
        x=[f"L{int(v)}" for v in bal['list_no']], y=bal['mean_position'],
        marker_color=[ACCENT_OK if abs(d) < 2 else ACCENT_NG for d in bal['편차']],
        hovertemplate="%{x}<br>평균 제시 위치 %{y:.1f}<extra></extra>"
    ))
    fig_b.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                        yaxis_title="평균 제시 위치", xaxis_title="리스트",
                        title=f"리스트별 평균 제시 위치 — 편차 폭 {oe['balance_spread']:.1f}")
    st.plotly_chart(fig_b)

# --- 심리측정 함수 ---
st.subheader("심리측정 함수 (SNR별 정답률)")
pf = (df_sel.groupby('snr_level', as_index=False)
      .agg(rate=('correct_rate', 'mean'), n=('correct_rate', 'size')))
fig_pf = go.Figure()
fig_pf.add_hline(y=50, line_dash="dot", line_color="#8b9b9e")
fig_pf.add_trace(go.Scatter(
    x=pf['snr_level'], y=pf['rate'] * 100, mode='lines+markers',
    line=dict(color=ACCENT_OK, width=2), marker=dict(size=9),
    hovertemplate="SNR %{x} dB<br>정답률 %{y:.1f}%<extra></extra>", name="전체"
))
fig_pf.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                     xaxis_title="SNR (dB)", yaxis_title="핵심어 정답률 (%)",
                     yaxis_range=[-5, 105], xaxis=dict(tickmode='array', tickvals=LADDER),
                     showlegend=False)
st.plotly_chart(fig_pf)

# ============================================================
# 5. 신뢰도 및 내보내기
# ============================================================
st.header("5. 신뢰도 및 내보내기")
rel = reliability(matrix_full)

if rel.get('status') == 'Success':
    r1, r2, r3 = st.columns(3)
    r1.metric("Cronbach α", f"{rel['alpha']:.3f}" if rel['alpha'] is not None else "–",
              help=f"리스트 {rel['k']}개를 문항으로 본 내적 일관성")
    r2.metric("단일 리스트 ICC",
              f"{anova['icc_single']:.3f}" if anova.get('icc_single') is not None else "–",
              help="리스트 1개만으로 개인 역치를 측정할 때의 신뢰도")
    r3.metric("측정오차 SD", f"{rel['error_sd']:.2f} dB",
              help=f"참가자·리스트 주효과를 제거한 오차항. 참가자 간 실력차는 별도 {rel['subject_sd']:.2f} dB")

    prec = pd.DataFrame([
        {'평균할 리스트 수': m, '95% 신뢰구간 폭 (±dB)': round(v, 2)}
        for m, v in rel['precision_95ci'].items()
    ])
    st.dataframe(prec, hide_index=True, width='stretch')
    st.caption(
        "개인 역치를 리스트 m개의 평균으로 측정할 때의 95% 신뢰구간 폭입니다. "
        "참가자 간 실력차를 제외한 순수 측정오차(sqrt(MS_error))를 기준으로 계산합니다."
    )
else:
    st.warning(f"신뢰도를 계산할 수 없습니다: {rel.get('status')}")

st.subheader("데이터 내보내기")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "문장 단위 (long) CSV",
    data=df_sel.drop(columns=['score', 'sentences']).to_csv(index=False).encode('utf-8-sig'),
    file_name="exp3_sentence_long.csv", mime="text/csv", width='stretch'
)
d2.download_button(
    "셀 단위 (참가자×리스트) CSV",
    data=cells.to_csv(index=False).encode('utf-8-sig'),
    file_name="exp3_cells.csv", mime="text/csv", width='stretch'
)
d3.download_button(
    "wide 행렬 CSV (JASP·SPSS)",
    data=matrix_full.to_csv().encode('utf-8-sig'),
    file_name="exp3_snr50_wide.csv", mime="text/csv", width='stretch'
)

with st.expander("분석 요약 텍스트"):
    lines = [
        f"실험 3 리스트 등가성 — 참가자 {len(sel_parts)}명, 리스트 {len(matrix_full.columns)}개",
        f"전체 평균 SNR-50: {grand_mean:+.2f} dB (K-QuickSIN 규준 {NORM_SNR50:+.2f} dB)",
        f"리스트 간 SNR-50 SD: {float(matrix_full.mean().std(ddof=1)):.2f} dB",
        f"기준 A(SD < {sd_limit:.1f} dB) 통과: {int(list_stats['pass_sd'].sum())}/{len(list_stats)}개",
        f"기준 B(±{dev_limit:.1f} dB) 통과: {int(list_stats['pass_dev'].sum())}/{len(list_stats)}개",
    ]
    if anova.get('status') == 'Success':
        lines.append(
            f"반복측정 ANOVA: F({anova['df_list']},{anova['df_err']})={anova['F']:.2f}, "
            f"p={anova['p']:.4f}, GG(eps={anova['eps_gg']:.2f}) p={anova['p_gg']:.4f}, "
            f"부분 eta2={anova['eta2_p']:.3f}"
        )
    if oe.get('status') == 'Success':
        lines.append(
            f"순서 효과: r={oe['r']:+.3f} "
            f"(참가자 수준 p={'–' if oe['p'] is None else format(oe['p'], '.4f')}), "
            f"18리스트 누적 {oe['total_db_over_lists']:+.2f} dB"
        )
    st.code("\n".join(lines), language=None)
