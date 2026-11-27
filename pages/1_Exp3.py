"""실험 3 — 최종 리스트 선정 결과 페이지.

18개 리스트 중 13개를 최종 리스트로 고른 과정과 근거를 DB 데이터로 다시 계산해 보여준다.
판정 규칙은 modules/exp3_selection.py에 고정되어 있다(화면에서 바꿀 수 없음).
"""

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from modules.auth import require_password
from modules.db_utils import init_supabase_client
from modules.exp3_utils import get_exp3_data
from modules import exp3_selection as S

st.set_page_config(page_title="실험 3 최종 리스트", layout="wide")
require_password()

# 색: 사용 = 파랑, 제외 = 주황 (색맹 검증 통과). 색만으로 구분하지 않도록 모양(●/✕)도 다르게 쓴다.
C_KEEP = "#1a73c8"
C_DROP = "#c0502a"
C_GRID = "rgba(128,128,128,0.25)"


def p_fmt(p: float) -> str:
    return "< .001" if p < .001 else f"{p:.3f}".lstrip("0") if p < 1 else "1.000"


# ============================================================
# 데이터 조회
# ============================================================
st.title("실험 3 — 최종 리스트 선정 결과")
st.markdown(
    "실험 3에서는 정상 청력 성인 18명이 **18개 리스트를 모두 들었고**(Williams 순서로 제시 순서 균형), "
    "리스트끼리 난이도가 같은지 확인해 **13개를 최종 리스트로 골랐습니다.** "
    "이 13개가 실험 4에 쓰였습니다. 아래 결과는 DB 데이터로 매번 다시 계산합니다."
)

supabase = init_supabase_client()
if not supabase:
    st.stop()

c1, c2 = st.columns([1, 5])
with c1:
    if st.button("🔄 데이터 새로고침", help="DB에서 다시 읽어옵니다(평소에는 10분간 저장된 결과를 씁니다)."):
        get_exp3_data.clear()
        st.session_state.pop("exp3_result", None)

if "exp3_result" not in st.session_state:
    with st.spinner("DB에서 실험 3 데이터를 읽고 판정을 다시 계산하는 중입니다..."):
        try:
            df = get_exp3_data(supabase)
        except Exception as e:
            st.error(f"데이터를 불러오지 못했습니다: {e}")
            st.stop()
        if df.empty:
            st.warning("실험 3 데이터가 없습니다.")
            st.stop()
        st.session_state.exp3_result = (df, S.run_all(df))

df, R = st.session_state.exp3_result
fi = R["final_info"]

# ============================================================
# 1. 결론
# ============================================================
st.header("1. 결론")
m = st.columns(5)
m[0].metric("최종 리스트", f"{len(R['kept'])}개")
m[1].metric("제외 리스트", f"{18 - len(R['kept'])}개", help="리스트 " + ", ".join(map(str, S.EXCLUDED_LISTS)))
m[2].metric("본분석 참가자", f"{R['n_main']}명", help=f"참가자 1~18번, {R['n_cells_main']}셀(18명 × 18리스트)")
m[3].metric("13개 리스트 효과", f"p = {p_fmt(fi['raw']['p_gg'])}",
            help="반복측정 ANOVA(Greenhouse-Geisser 보정). p ≥ .05면 리스트 간 난이도 차이가 없다는 뜻입니다.")
m[4].metric("13개 평균 SNR-50", f"{R['mean13']:.2f} dB", help=f"18개 전체 평균은 {R['mean18']:.2f} dB")

st.success(
    f"**최종 13개 리스트: {', '.join(map(str, S.FINAL_LISTS))}**  \n"
    f"제외: {', '.join(map(str, S.EXCLUDED_LISTS))}. "
    f"남은 13개는 리스트 간 난이도 차이가 없었고(p = {p_fmt(fi['raw']['p_gg'])}, 순서 보정 시 "
    f"p = {p_fmt(fi['adj']['p_gg'])}), 모두 평균에서 ±1.0 dB 안에 있었습니다(동등성 검정 통과)."
)
if not R["matches_final"]:
    st.error(
        f"⚠️ DB 데이터로 다시 계산한 결과({R['kept']})가 확정 결과와 다릅니다. "
        "실험 3 데이터가 수정되었을 수 있습니다. 개발 담당자에게 확인을 요청하세요."
    )
else:
    st.caption("✔ DB 데이터로 다시 계산한 결과가 2026-11-17 확정 결과와 같습니다.")

# ============================================================
# 2. 왜 13개인가
# ============================================================
st.header("2. 왜 13개만 남았나")
st.markdown(
    "각 리스트의 점수는 **SNR-50**(문장 키워드의 절반을 알아듣는 신호대잡음비)으로 바꿔 비교했습니다. "
    "값이 **낮을수록 쉬운 리스트**입니다. 리스트가 서로 바꿔 써도 될 만큼 비슷해야 하므로, "
    "평균에서 많이 벗어난 리스트를 아래 기준으로 걸러냈습니다."
)

# 2-1. 그래프: 18개 리스트 평균
v = R["verdict"].copy()
main = R["main"]
g = main.groupby("list_no")["snr50"]
n = R["n_main"]
v["ci"] = v["list_no"].map(stats.t.ppf(.975, n - 1) * g.std(ddof=1) / np.sqrt(n))
v = v.sort_values("raw_mean")
fig = go.Figure()
for label, color, symbol in [("사용", C_KEEP, "circle"), ("제외", C_DROP, "x")]:
    d = v[v["판정"] == label]
    fig.add_trace(go.Scatter(
        x=d["raw_mean"], y=[f"리스트 {l}" for l in d["list_no"]], mode="markers", name=f"{label} ({len(d)}개)",
        marker=dict(color=color, size=11, symbol=symbol, line=dict(width=0)),
        error_x=dict(type="data", array=d["ci"], color=color, thickness=2, width=0),
        customdata=np.stack([d["판정"], d["사유"], d["ci"]], axis=-1),
        hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>평균 SNR-50 %{x:.2f} dB (95% CI ±%{customdata[2]:.2f})"
                      "<br>%{customdata[1]}<extra></extra>",
    ))
fig.add_vline(x=R["mean18"], line=dict(color="gray", dash="dot", width=1.5),
              annotation_text=f"18개 평균 {R['mean18']:.2f} dB", annotation_position="top")
fig.update_layout(
    title="리스트별 평균 SNR-50 (참가자 18명, 막대는 95% 신뢰구간)", height=560,
    xaxis=dict(title="SNR-50 (dB) — 왼쪽일수록 쉬움", gridcolor=C_GRID, zeroline=False),
    yaxis=dict(categoryorder="array", categoryarray=[f"리스트 {l}" for l in v["list_no"]], autorange="reversed"),
    legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=10, r=10, t=80, b=40), hovermode="closest",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "리스트 2는 다른 리스트보다 뚜렷하게 쉬웠고(−1.7 dB), 리스트 1·17도 쉬운 쪽에 치우쳤습니다. "
    "리스트 8·9는 평균에 가깝지만 참가자 간 편차가 커서 '같다'고 확신할 수 없었습니다(동등성 검정 미달)."
)

# 2-2. 기준
st.subheader("판정 기준")
st.dataframe(S.CRITERIA, hide_index=True, width="stretch")
st.markdown(
    "- **A 또는 B에 해당 → 제외.** C·D·E 중 2개 이상 → 제외. 1개만 해당하면 '주의' 표시 후 유지.\n"
    "- 리스트를 빼면 전체 평균(기준점)이 움직이므로, **남은 리스트로 같은 기준을 다시 적용**했습니다(단계적 제외).\n"
    "- **최종 확인:** 남은 리스트 전부 동등성 검정(C) 통과 + 리스트 효과 비유의 + 12개 이상.\n"
    "- 제시 순서가 뒤로 갈수록 점수가 좋아지는 효과(학습 효과)가 있어, 편차는 **순서 효과를 보정한 값**으로 계산했습니다."
)

# 2-3. 단계
st.subheader("판정 과정 (4단계)")
stage_tbl = pd.DataFrame([{
    "단계": s["stage"],
    "대상": f"{s['n_lists']}개",
    "제외": ", ".join(map(str, s["excluded"])) or "없음 (전 기준 통과)",
    "사유": s["reason"] or "—",
    "리스트 효과 p (원자료)": p_fmt(s["gg_raw"]),
    "리스트 효과 p (순서 보정)": p_fmt(s["gg_adj"]),
} for s in R["stages"]])
st.dataframe(stage_tbl, hide_index=True, width="stretch")
st.caption("p는 반복측정 ANOVA(Greenhouse-Geisser 보정). 1단계 18개에서는 p < .05로 리스트 간 차이가 있었고, "
           "4단계 13개에서는 차이가 없었습니다.")

# 2-4. 최종 13개 동등성 그래프
fr = R["final_rows"].sort_values("dev")
fig2 = go.Figure()
fig2.add_vrect(x0=-S.EQ_MARGIN, x1=S.EQ_MARGIN, fillcolor="rgba(26,115,200,0.08)", line_width=0)
fig2.add_trace(go.Scatter(
    x=fr["dev"], y=[f"리스트 {l}" for l in fr["list_no"]], mode="markers", name="편차와 90% 신뢰구간",
    marker=dict(color=C_KEEP, size=10),
    error_x=dict(type="data", symmetric=False, array=fr["ci_hi"] - fr["dev"], arrayminus=fr["dev"] - fr["ci_lo"],
                 color=C_KEEP, thickness=2, width=0),
    customdata=np.stack([fr["ci_lo"], fr["ci_hi"]], axis=-1),
    hovertemplate="<b>%{y}</b><br>편차 %{x:+.2f} dB<br>90% CI [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]<extra></extra>",
))
for x in (-S.EQ_MARGIN, S.EQ_MARGIN):
    fig2.add_vline(x=x, line=dict(color="gray", dash="dash", width=1))
fig2.update_layout(
    title="최종 13개: 평균과의 차이(순서 보정)가 모두 ±1.0 dB 허용 범위 안", height=460, showlegend=False,
    xaxis=dict(title="13개 평균과의 차이 (dB)", range=[-1.6, 1.6], gridcolor=C_GRID, zeroline=True,
               zerolinecolor="gray"),
    yaxis=dict(categoryorder="array", categoryarray=[f"리스트 {l}" for l in fr["list_no"]], autorange="reversed"),
    margin=dict(l=10, r=10, t=60, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 3. 리스트별 판정표
# ============================================================
st.header("3. 리스트별 판정표")
st.caption("편차는 SNR-50 기준입니다. 음수는 평균보다 쉬운 리스트, 양수는 어려운 리스트입니다.")
vt = R["verdict"].copy()
vt["13개 기준 편차 [90% CI]"] = vt.apply(
    lambda r: "—" if pd.isna(r.dev13) else f"{r.dev13:+.2f} [{r.lo13:+.2f}, {r.hi13:+.2f}]", axis=1)
vt = vt.rename(columns={"list_no": "리스트", "raw_mean": "평균 SNR-50 (dB)",
                        "dev18": "18개 기준 편차 (dB)", "holm18": "18개 기준 Holm p"})
st.dataframe(
    vt[["리스트", "판정", "사유", "평균 SNR-50 (dB)", "18개 기준 편차 (dB)", "18개 기준 Holm p", "13개 기준 편차 [90% CI]"]],
    hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%.2f")
                   for c in ["평균 SNR-50 (dB)", "18개 기준 편차 (dB)", "18개 기준 Holm p"]},
)

# ============================================================
# 4. 문장 점검
# ============================================================
st.header("4. 문장 점검 (기준 E)")
st.markdown(
    "문장 108개는 각각 정해진 SNR 하나에서만 들려줬으므로, **같은 SNR의 문장 18개끼리** 비교했습니다. "
    "실질적인 문제 문장은 **리스트 2의 −5 dB 문장 하나**였습니다."
)
sf = R["sent"][R["sent"]["z_flag"]].copy()
sf["판정"] = np.where(sf["E"], "해당", np.where(sf["snr_level"] > 0, "비해당 (천장효과)", "비해당 (바닥효과)"))
sf = sf.rename(columns={"list_no": "리스트", "snr_level": "SNR (dB)", "mean_correct": "문장 평균 (키워드)",
                        "snr_mean": "같은 SNR 평균", "z": "z", "diff": "실제 차이"})
st.dataframe(sf[["리스트", "SNR (dB)", "문장 평균 (키워드)", "같은 SNR 평균", "z", "실제 차이", "판정"]],
             hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.2f")
                            for c in ["문장 평균 (키워드)", "같은 SNR 평균", "z", "실제 차이"]})
st.caption(
    "+4 dB와 −11 dB에서는 거의 모든 문장이 만점이거나 0점이라 문장 간 차이가 아주 작고 z가 부풀려집니다. "
    "그래서 '실제 차이 ≥ 0.5 키워드' 조건을 함께 봤습니다."
)

# ============================================================
# 5. 순서 효과
# ============================================================
st.header("5. 제시 순서 효과")
pos = R["position"]
fig3 = go.Figure(go.Scatter(
    x=pos["presentation_order"], y=pos["mean"], mode="lines+markers", line=dict(color=C_KEEP, width=2),
    marker=dict(size=8), name="평균 SNR-50",
    hovertemplate="%{x}번째 리스트<br>평균 SNR-50 %{y:.2f} dB<extra></extra>",
))
fig3.update_layout(
    title="몇 번째로 들은 리스트인지에 따른 평균 SNR-50 (18리스트 전체)", height=360, showlegend=False,
    xaxis=dict(title="제시 순서", dtick=1, gridcolor=C_GRID),
    yaxis=dict(title="SNR-50 (dB)", gridcolor=C_GRID), margin=dict(l=10, r=10, t=60, b=40),
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    f"뒤로 갈수록 리스트 1개당 평균 {abs(R['slope'].slope):.2f} dB씩 좋아졌습니다(학습 효과). "
    "모든 리스트가 각 순서에 한 번씩 오도록 설계(Williams)했기 때문에 리스트 비교에는 영향이 없고, "
    "편차 계산에서는 이 효과를 보정했습니다."
)

# ============================================================
# 6. 민감도 분석
# ============================================================
st.header("6. 민감도 분석")
st.markdown(
    "조건을 바꿔도 결론이 유지되는지 확인했습니다. **리스트 2, 1, 17 제외는 모든 조건에서 같았습니다.** "
    "리스트 8·9는 조건에 따라 판정이 바뀌는 경계선이었고, 최종 13개는 대부분 조건에서 동등성을 통과했습니다."
)
sens = R["sensitivity"].copy()
for c in ["13개 리스트 효과 GG p (원자료)", "13개 리스트 효과 GG p (순서 보정)"]:
    sens[c] = sens[c].map(p_fmt)
st.dataframe(sens, hide_index=True, width="stretch")
st.caption("②에서 리스트 14는 신뢰구간이 1.01 dB로 허용 범위를 아주 조금 넘었습니다. "
           "④는 사전 기준 그대로 적용한 경우로, 리스트 3(−11 dB 문장, 실제 차이 0.24 키워드)이 추가로 빠져 12개가 됩니다.")

# ============================================================
# 7. 보고 시 명시할 사항
# ============================================================
st.header("7. 보고서·논문에 밝혀야 할 점")
st.markdown(
    "1. **단계적 제외**: 최종 확인을 통과하지 못했을 때의 처리는 사전에 정하지 않았습니다. "
    "같은 기준을 남은 리스트에 반복 적용했고 4단계에서 멈췄습니다.\n"
    "2. **기준 E의 '실제 차이 ≥ 0.5 키워드' 조건**: 판정 후 천장·바닥 구간에서 z가 부풀려지는 문제를 확인하고 추가했습니다. "
    "사전 기준대로라면 리스트 3도 제외되어 12개가 됩니다(민감도 분석 ④).\n"
    "3. **결론 범위**: '정상 청력 성인 18명에서의 리스트 동등성'입니다. 난청군에서의 동등성은 후속 검증이 필요합니다."
)

# ============================================================
# 8. 데이터 다운로드
# ============================================================
st.header("8. 데이터 다운로드")
st.markdown(
    "엑셀에서 바로 열 수 있는 CSV 파일 묶음(ZIP)입니다. **이름 등 개인정보는 들어 있지 않고 참가자 번호만 있습니다.** "
    "검사 문장 원문은 검사 자료 보호를 위해 빼고 문장 번호만 넣었습니다."
)

CODEBOOK = pd.DataFrame([
    ("exp3_cells.csv", "participant_no", "참가자 번호 (1~18 본분석, 32~34 보조)"),
    ("exp3_cells.csv", "group", "본분석 / 보조"),
    ("exp3_cells.csv", "list_no", "리스트 번호 (1~18)"),
    ("exp3_cells.csv", "presentation_order", "제시 순서 (1~18번째)"),
    ("exp3_cells.csv", "total_correct", "리스트 정답 키워드 수 (0~30)"),
    ("exp3_cells.csv", "snr50", "SNR-50 (dB) = 5.5 − 0.6 × total_correct"),
    ("exp3_cells.csv", "final_list", "최종 13개 리스트 여부"),
    ("exp3_sentences.csv", "trial_index", "리스트 안 문장 순서"),
    ("exp3_sentences.csv", "snr_level", "문장 SNR (dB): +4, +1, −2, −5, −8, −11"),
    ("exp3_sentences.csv", "sentence_id", "문장 번호"),
    ("exp3_sentences.csv", "keywords_correct", "맞힌 키워드 수 (0~5)"),
    ("exp3_list_verdicts.csv", "판정/사유", "리스트별 최종 판정과 제외 사유"),
    ("exp3_list_verdicts.csv", "dev18, holm18", "18개 기준 순서 보정 편차(dB), Holm 보정 p"),
    ("exp3_list_verdicts.csv", "dev13, lo13, hi13", "13개 기준 편차와 90% 신뢰구간(dB)"),
    ("exp3_stages.csv", "—", "단계별 대상 리스트·제외 리스트·사유·리스트 효과 p"),
    ("exp3_sentence_check.csv", "z, E", "같은 SNR 문장 간 z, 기준 E 해당 여부"),
    ("exp3_sensitivity.csv", "—", "민감도 분석 조건별 결과"),
], columns=["파일", "열", "설명"])


def build_zip() -> bytes:
    cells = R["cells"].copy()
    cells["group"] = np.where(cells["participant_no"].isin(S.MAIN_PARTICIPANTS), "본분석", "보조")
    cells["final_list"] = cells["list_no"].isin(S.FINAL_LISTS)
    cells["snr50"] = cells["snr50"].round(2)
    cells = cells[["participant_no", "group", "list_no", "presentation_order", "total_correct", "snr50", "final_list"]]

    sent = df[["participant_no", "list_no", "presentation_order", "trial_index", "snr_level", "sentence_id",
               "total_score", "num_keywords"]].rename(columns={"total_score": "keywords_correct"}).copy()
    sent["final_list"] = sent["list_no"].isin(S.FINAL_LISTS)

    stages = pd.DataFrame([{
        "stage": s["stage"], "n_lists": s["n_lists"], "lists": " ".join(map(str, s["lists"])),
        "excluded": " ".join(map(str, s["excluded"])), "reason": s["reason"],
        "list_effect_p_gg_raw": round(s["gg_raw"], 4), "list_effect_p_gg_order_adjusted": round(s["gg_adj"], 4),
    } for s in R["stages"]])

    readme = (
        "Quick-SIN 한국어판 실험 3 데이터\n"
        f"내려받은 시각: {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"최종 13개 리스트: {', '.join(map(str, S.FINAL_LISTS))}\n"
        f"제외 리스트: {', '.join(map(str, S.EXCLUDED_LISTS))}\n"
        "본분석: 참가자 1~18번 (Williams 18x18, 324셀). 보조 32~34번은 민감도 분석에만 사용.\n"
        "SNR-50 = 5.5 - 0.6 x 정답 키워드 수 (사다리 +4 ~ -11 dB, 3 dB 간격, 문장당 키워드 5개)\n"
        "개인정보(이름 등)와 검사 문장 원문은 포함하지 않음.\n"
        "각 열의 뜻은 codebook.csv 참고.\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name, frame):
            z.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))  # 엑셀 한글 깨짐 방지
        put("exp3_cells.csv", cells)
        put("exp3_sentences.csv", sent)
        put("exp3_list_verdicts.csv", R["verdict"].round(4))
        put("exp3_stages.csv", stages)
        put("exp3_sentence_check.csv", R["sent"].round(4))
        put("exp3_sensitivity.csv", R["sensitivity"].round(4))
        put("codebook.csv", CODEBOOK)
        z.writestr("README.txt", readme.encode("utf-8-sig"))
    return buf.getvalue()


st.download_button("📦 실험 3 데이터 전체 내려받기 (ZIP)", data=build_zip(),
                   file_name=f"qsin_exp3_data_{datetime.now():%Y%m%d}.zip", mime="application/zip", type="primary")
with st.expander("ZIP 안에 들어 있는 파일과 열 설명"):
    st.dataframe(CODEBOOK, hide_index=True, width="stretch")

# ============================================================
# 개발자 참고
# ============================================================
with st.expander("개발자 참고 (계산 방법)"):
    st.markdown(
        "- 계산 코드: `modules/exp3_selection.py`. 판정 기준값과 확정 결과(`FINAL_LISTS`)가 상수로 고정되어 있습니다.\n"
        "- 데이터: `test_reports_qsin.experiment = 'exp3'` × `score_qsin`. 리스트 번호는 `sound_set`, "
        "제시 순서는 `presentation_order`, 문장 SNR은 `score_qsin.snr_level`.\n"
        "- 구버전 앱 복제 버그 세션 6개는 `modules/exp3_utils.EXCLUDED_SESSIONS`로 제외합니다(DB에는 보존).\n"
        "- 셀 = 참가자 × 리스트. 사다리 6문장이 모두 있는 셀만 사용.\n"
        "- 순서 보정: OLS `SNR-50 ~ 참가자 + 제시순서 + 리스트`로 순서 효과를 추정해 뺀 값으로 편차·TOST·잔차 SD를 계산.\n"
        "- 편차 = 참가자별로 (리스트 값 − 그 참가자의 리스트 평균)의 평균. B는 이 편차의 단일표본 t검정 + Holm 보정.\n"
        "- 페이지를 열 때 다시 계산한 결과가 `FINAL_LISTS`와 다르면 빨간 경고가 뜹니다."
    )
