import streamlit as st
import pandas as pd
import numpy as np
from modules.db_utils import init_supabase_client, get_patients
from modules.analysis_utils import (
    get_all_sentence_data,
    analyze_all_sentences,
    analyze_subjects,
    TARGET_SNR50,
)

st.set_page_config(page_title="피험자 선택 분석", layout="wide")
st.title("Quick-SIN 피험자 선택 분석 페이지 (Selector)")
st.write(
    "분석에 포함할 피험자를 직접 선택하고, 선택한 피험자들의 데이터만으로 "
    "SNR-50을 추정하여 최대·최소·중앙값·평균·표준편차 등의 통계를 확인합니다."
)


def _is_hearing_loss(series: pd.Series) -> pd.Series:
    """hearing_loss 컬럼을 bool/str/None 모두 안전하게 처리하여 boolean mask 반환"""
    return series.map(lambda v: str(v).strip().lower() == 'true' if pd.notna(v) else False)


def _summary_stats(values: pd.Series) -> dict:
    """SNR-50 등 수치 시리즈의 요약 통계를 계산한다."""
    clean = pd.to_numeric(values, errors='coerce').dropna()
    if clean.empty:
        return {k: None for k in ['count', 'max', 'min', 'median', 'mean', 'std']}
    return {
        'count': int(clean.count()),
        'max': float(clean.max()),
        'min': float(clean.min()),
        'median': float(clean.median()),
        'mean': float(clean.mean()),
        'std': float(clean.std()) if clean.count() > 1 else 0.0,
    }


def _render_summary_metrics(stats: dict, unit: str = "dB"):
    """요약 통계를 st.metric 6분할로 표시한다."""
    cols = st.columns(6)
    labels = [
        ('count', "데이터 수", ""),
        ('max', "최대값", unit),
        ('min', "최소값", unit),
        ('median', "중앙값", unit),
        ('mean', "평균값", unit),
        ('std', "표준편차", unit),
    ]
    for col, (key, label, u) in zip(cols, labels):
        with col:
            val = stats.get(key)
            if val is None:
                col.metric(label, "-")
            elif key == 'count':
                col.metric(label, f"{val:,}")
            else:
                col.metric(label, f"{val:.2f} {u}".strip())


# --- 클라이언트 초기화 ---
supabase = init_supabase_client()
if not supabase:
    st.stop()

# --- 세션 상태 초기화 ---
for key in ['sel_raw_df', 'sel_subject_df', 'sel_sentence_df', 'sel_snr_range']:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# 섹션 0: 데이터 범위 및 피험자 선택
# ============================================================
st.header("1. 데이터 조회 및 피험자 선택")

use_dummy = st.checkbox(
    "테스트 데이터(dummy_ 접두사)만 사용",
    value=False,
    key='selector_dummy_check',
    help="체크 시 session_id가 'dummy_'로 시작하는 데이터만 사용합니다. 체크 해제 시 그 외의 데이터를 사용합니다."
)

if st.button("전체 데이터 조회 (피험자 목록 불러오기)"):
    with st.spinner("전체 데이터를 DB에서 조회 중입니다..."):
        all_data_df = get_all_sentence_data(supabase, use_dummy_prefix=use_dummy)
        if all_data_df.empty:
            st.session_state.sel_raw_df = None
            st.warning("조회된 데이터가 없습니다.")
        else:
            st.session_state.sel_raw_df = all_data_df
            st.session_state.sel_snr_range = (
                all_data_df['snr_level'].min(),
                all_data_df['snr_level'].max()
            )
            # 새 데이터 조회 시 이전 분석 결과 초기화
            st.session_state.sel_subject_df = None
            st.session_state.sel_sentence_df = None
            st.success(f"데이터 조회 완료 ({len(all_data_df):,}행).")

raw_df = st.session_state.sel_raw_df

if raw_df is not None and not raw_df.empty:
    # 데이터가 실제로 존재하는 피험자만 선택지로 노출
    patients_meta = get_patients(supabase)
    name_map = {p['id']: p['name'] for p in patients_meta} if patients_meta else {}

    # patient_user_id 별 데이터 행 수 / 난청 여부 집계
    hl_mask = _is_hearing_loss(raw_df['hearing_loss'])
    raw_df = raw_df.assign(_hl=hl_mask)

    subj_info = (
        raw_df.dropna(subset=['patient_user_id'])
        .groupby('patient_user_id')
        .agg(rows=('sentence_id', 'size'),
             sentences=('sentence_id', 'nunique'),
             snr_levels=('snr_level', 'nunique'),
             hearing_loss=('_hl', 'max'))
        .reset_index()
    )

    if subj_info.empty:
        st.warning("patient_user_id가 연결된 데이터가 없어 피험자를 선택할 수 없습니다.")
        st.stop()

    subj_info['name'] = subj_info['patient_user_id'].map(
        lambda pid: name_map.get(pid, "(이름 미상)")
    )
    subj_info['group'] = subj_info['hearing_loss'].map(lambda x: "난청군" if x else "정상군")
    # 라벨: 이름 + 그룹 + 데이터 수 (id로 중복 방지)
    subj_info['label'] = subj_info.apply(
        lambda r: f"{r['name']} [{r['group']}] · {int(r['rows'])}행 ({r['patient_user_id'][:8]})",
        axis=1
    )

    st.caption(f"데이터가 있는 피험자: 총 {len(subj_info)}명 "
               f"(정상군 {int((~subj_info['hearing_loss']).sum())}명, "
               f"난청군 {int(subj_info['hearing_loss'].sum())}명)")

    # 그룹 필터
    group_filter = st.radio(
        "그룹 필터", ["전체", "정상군만", "난청군만"], index=0, horizontal=True
    )
    filtered_info = subj_info
    if group_filter == "정상군만":
        filtered_info = subj_info[~subj_info['hearing_loss']]
    elif group_filter == "난청군만":
        filtered_info = subj_info[subj_info['hearing_loss']]

    label_to_id = dict(zip(filtered_info['label'], filtered_info['patient_user_id']))
    options = list(label_to_id.keys())

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        selected_labels = st.multiselect(
            "분석할 피험자 선택",
            options=options,
            help="여러 명을 선택할 수 있습니다."
        )
    with col_btn:
        st.write("")
        st.write("")
        select_all = st.checkbox("전체 선택", value=False)

    if select_all:
        selected_labels = options

    selected_ids = [label_to_id[lbl] for lbl in selected_labels]

    st.divider()

    # ============================================================
    # 섹션 1: 선택한 피험자 분석
    # ============================================================
    st.header("2. 선택한 피험자 분석")

    if not selected_ids:
        st.info("위에서 분석할 피험자를 한 명 이상 선택해주세요.")
    else:
        st.write(f"선택된 피험자: **{len(selected_ids)}명**")

        if st.button("선택한 피험자 분석 실행", type="primary"):
            subset = raw_df[raw_df['patient_user_id'].isin(selected_ids)].drop(columns=['_hl'])

            with st.spinner("피험자별 SNR-50 분석 중..."):
                st.session_state.sel_subject_df = analyze_subjects(subset)
            with st.spinner("문장별 SNR-50 분석 중..."):
                st.session_state.sel_sentence_df = analyze_all_sentences(subset, label="[선택] ")

        subject_df = st.session_state.sel_subject_df
        sentence_df = st.session_state.sel_sentence_df

        if subject_df is not None or sentence_df is not None:
            tab_subj, tab_sent = st.tabs(["피험자별 SNR-50", "문장별 SNR-50"])

            # ----------------------------------------------------
            # 피험자별 탭
            # ----------------------------------------------------
            with tab_subj:
                if subject_df is None or subject_df.empty:
                    st.warning("피험자별 분석 결과가 없습니다. (데이터 부족 또는 정답률 불변)")
                else:
                    # Extrapolated 제외한 유효 피험자 기준 통계
                    valid = subject_df[subject_df.get('validity', 'Analyzed') != 'Extrapolated'] \
                        if 'validity' in subject_df.columns else subject_df
                    extrapolated_count = len(subject_df) - len(valid)

                    st.subheader("SNR-50 요약 통계 (유효 피험자 기준)")
                    _render_summary_metrics(_summary_stats(valid['snr_50']))

                    if extrapolated_count > 0:
                        st.caption(f"* {extrapolated_count}명은 Extrapolated(신뢰도 낮음)로 통계에서 제외되었습니다.")

                    st.subheader("피험자별 결과 테이블")
                    tbl = subject_df.copy()
                    for c in ['snr_50', 'snr_loss', 'slope']:
                        if c in tbl.columns:
                            tbl[c] = tbl[c].round(2)
                    tbl['name'] = tbl['patient_user_id'].map(lambda pid: name_map.get(pid, "(이름 미상)"))
                    st.dataframe(
                        tbl,
                        column_config={
                            "patient_user_id": "피험자 ID",
                            "name": "이름",
                            "snr_50": "SNR-50 (dB)",
                            "snr_loss": f"SNR Loss (dB, vs {TARGET_SNR50})",
                            "grade": "등급",
                            "validity": "신뢰도",
                            "slope": "기울기 (%/dB)",
                            "data_points": "데이터 수",
                            "sentences_tested": "테스트 문장 수",
                        },
                        use_container_width=True
                    )

                    csv = subject_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "피험자별 분석 결과 다운로드 (CSV)",
                        data=csv,
                        file_name="qsin_selected_subjects.csv",
                        mime="text/csv",
                    )

            # ----------------------------------------------------
            # 문장별 탭
            # ----------------------------------------------------
            with tab_sent:
                if sentence_df is None or sentence_df.empty:
                    st.warning("문장별 분석 결과가 없습니다. (데이터 부족 또는 정답률 불변)")
                else:
                    valid_s = sentence_df[sentence_df.get('validity', 'Analyzed') != 'Extrapolated'] \
                        if 'validity' in sentence_df.columns else sentence_df
                    extrapolated_s = len(sentence_df) - len(valid_s)

                    st.subheader("SNR-50 요약 통계 (유효 문장 기준)")
                    _render_summary_metrics(_summary_stats(valid_s['snr_50']))

                    if extrapolated_s > 0:
                        st.caption(f"* {extrapolated_s}개 문장은 Extrapolated(신뢰도 낮음)로 통계에서 제외되었습니다.")

                    st.subheader("문장별 결과 테이블")
                    tbl_s = sentence_df.copy()
                    for c in ['snr_50', 'slope', 'avg_score']:
                        if c in tbl_s.columns:
                            tbl_s[c] = tbl_s[c].round(2)
                    if 'total_score_sum' in tbl_s.columns:
                        tbl_s['total_score_sum'] = tbl_s['total_score_sum'].round(0)
                    st.dataframe(
                        tbl_s,
                        column_config={
                            "sentence_id": "문장 ID",
                            "full_sentence": st.column_config.TextColumn("문장", width="large"),
                            "snr_50": "SNR-50 (dB)",
                            "slope": "기울기 (%/dB)",
                            "validity": "신뢰도",
                            "total_score_sum": "총 점수",
                            "avg_score": "평균 점수",
                            "data_points": "데이터 수",
                            "snr_levels": "SNR 레벨 수",
                        },
                        use_container_width=True
                    )

                    csv_s = sentence_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "문장별 분석 결과 다운로드 (CSV)",
                        data=csv_s,
                        file_name="qsin_selected_sentences.csv",
                        mime="text/csv",
                    )
else:
    st.info("먼저 위의 **전체 데이터 조회** 버튼을 눌러 피험자 목록을 불러오세요.")
