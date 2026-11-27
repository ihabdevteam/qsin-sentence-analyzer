"""실험 4 Word(.docx) 보고서 자동 생성.

페이지의 '보고서 내려받기' 버튼에서 호출한다. 숫자·표·그래프는 매번 DB 데이터로 다시 계산하고,
설명 문장은 아래 틀에 숫자만 바꿔 넣는다. 개인정보(이름 등)는 넣지 않고 참가자 번호만 쓴다.

그래프는 matplotlib으로 그린다. 한글 글꼴(맑은 고딕/나눔고딕)이 없는 환경이면 그래프 글자만 영어로 바뀐다
(Streamlit Cloud는 packages.txt의 fonts-nanum으로 나눔고딕을 설치한다).
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from modules import exp4_utils as U  # noqa: E402

C1, C2, GRAY = "#1a73c8", "#c0502a", "#888888"
FONT_DOC = "맑은 고딕"


# ------------------------------------------------------------
# 글꼴
# ------------------------------------------------------------
def _setup_plot_font() -> bool:
    """한글 글꼴이 있으면 쓰고 True, 없으면 False(그래프 글자를 영어로)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic", "Noto Sans CJK KR"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


KO = _setup_plot_font()


def _t(ko: str, en: str) -> str:
    return ko if KO else en


def p_fmt(p: float) -> str:
    return "< .001" if p < .001 else f"= {p:.3f}".replace("0.", ".")


# ------------------------------------------------------------
# docx 도우미
# ------------------------------------------------------------
def _style(doc: Document):
    st = doc.styles["Normal"]
    st.font.name = FONT_DOC
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_DOC)
    for name in ["Heading 1", "Heading 2", "Heading 3", "Title"]:
        s = doc.styles[name]
        s.font.name = FONT_DOC
        s.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_DOC)
        s.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.0)


def _para(doc, text, bold=False, size=None, color=None, italic=False, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
    if align:
        p.alignment = align
    return p


def _bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def _table(doc, df: pd.DataFrame, widths=None, font=9):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(font)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = "" if (isinstance(v, float) and np.isnan(v)) else str(v)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def _figure(doc, fig, width_cm=15.5, caption=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    doc.add_picture(buf, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        _para(doc, caption, size=9, color="#555555", align=WD_ALIGN_PARAGRAPH.CENTER)


def _ax_style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------
# 그래프
# ------------------------------------------------------------
def _fig_lists(e13: dict, final_lists):
    ls = e13["lists"].sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for _, r in ls.iterrows():
        keep = int(r.list_no) in final_lists
        c = C1 if keep else C2
        y = _t(f"리스트 {int(r.list_no)}", f"List {int(r.list_no)}")
        ax.errorbar(r["mean"], y, xerr=r["ci"], fmt="o" if keep else "X", color=c, ms=7, lw=2, capsize=0)
    ax.axvline(e13["grand"], color=GRAY, ls=":", lw=1.2)
    ax.invert_yaxis()
    ax.set_xlabel(_t("SNR-50 (dB), 왼쪽일수록 쉬움", "SNR-50 (dB), left = easier"))
    ax.set_title(_t("리스트별 평균 SNR-50 (1차, 95% CI) · × = 제외", "Mean SNR-50 by list (round 1, 95% CI) · X = excluded"),
                 fontsize=10)
    _ax_style(ax)
    return fig


def _fig_position(pe: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for r, c, ls_, lab in [(1, C1, "-", _t("1차", "Round 1")), (2, C2, "--", _t("2차", "Round 2"))]:
        d = pe[pe["test_round"] == r]
        if len(d):
            ax.plot(d["presentation_order"], d["mean"], ls_, marker="o", color=c, lw=2, ms=5,
                    label=f"{lab} (n={int(d['count'].max())})")
    ax.set_xticks(range(1, 14))
    ax.set_xlabel(_t("제시 순서", "Presentation order"))
    ax.set_ylabel("SNR-50 (dB)")
    ax.legend(frameon=False)
    ax.set_title(_t("제시 순서별 평균 SNR-50", "Mean SNR-50 by presentation order"), fontsize=10)
    _ax_style(ax)
    return fig


def _fig_retest(rt: dict):
    pr, ml = rt["pairs"], rt["mean_level"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    lo, hi = min(pr.r1.min(), pr.r2.min()) - .5, max(pr.r1.max(), pr.r2.max()) + .5
    ax = axes[0]
    ax.plot([lo, hi], [lo, hi], ":", color=GRAY, lw=1)
    ax.scatter(pr.r1, pr.r2, color=C1, s=28)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(_t("1차 (dB)", "Round 1 (dB)"))
    ax.set_ylabel(_t("2차 (dB)", "Round 2 (dB)"))
    ax.set_title(_t("참가자별 12개 평균", "Participant 12-list mean"), fontsize=10)
    _ax_style(ax)
    ax = axes[1]
    avg, dif = (pr.r1 + pr.r2) / 2, pr.r2 - pr.r1
    ax.scatter(avg, dif, color=C1, s=28)
    ax.axhline(ml["diff_mean"], color=GRAY, lw=1.2)
    for v in ml["loa"]:
        ax.axhline(v, color=C2, ls="--", lw=1)
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel(_t("1·2차 평균 (dB)", "Mean of rounds (dB)"))
    ax.set_ylabel(_t("2차 - 1차 (dB)", "Round 2 - 1 (dB)"))
    ax.set_title("Bland-Altman", fontsize=10)
    _ax_style(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# 고정 내용
# ------------------------------------------------------------
HISTORY = pd.DataFrame([
    ("실험 1", "문장–소음 조합 SNR-50 산출", "정상 15명, 문장 360개 × SNR 3개(−7, −4, −2 dB), 4명 화자 babble",
     "신뢰도 높은 문장 253개(70.3%) 중 SNR-50 −5 ~ −9 dB인 186개 선별"),
    ("실험 2-1-1", "SNR-50 정밀 재산출", "정상 12명, 186문장, 문장당 SNR 7개로 심리음향곡선 전 범위 탐색",
     "평균 SNR-50 −4.16 dB(중앙값 −4.20), 약 85% 적합"),
    ("실험 2-1-2", "소음 재보정 후 균등성 검증", "정상 3명, 목표 −4 dB로 문장별 소음 레벨 조정, SNR 5개",
     "158문장 SD 0.69 dB → 3단계 선별로 112문장(평균 −3.66, SD 0.38 dB)"),
    ("실험 3", "리스트 동등성(18개 리스트)", "정상 18명, 리스트 18개 × 6문장(+4 ~ −11 dB), Williams 18×18",
     "기준 A~E 단계 적용으로 5개 제외, 13개 리스트 선정(리스트 효과 GG p = .71)"),
    ("실험 4", "최종 검증 + 검사-재검사 + 정상 기준", "정상 26명, 13개 리스트, Williams 26행, 1~2주 뒤 재검사",
     "본 보고서"),
], columns=["실험", "목적", "방법", "결과"])


# ------------------------------------------------------------
# 보고서 본체
# ------------------------------------------------------------
def build_report(df: pd.DataFrame, n_excluded: int | None = None) -> bytes:
    cells = U.build_cells(df)
    status = U.collection_status(cells)
    stg = U.stage(status)
    m13 = U.matrix(cells, 1, U.LISTS)
    e13 = U.equivalence(m13)
    e12 = U.equivalence(U.matrix(cells, 1, U.FINAL_LISTS))
    fd = U.final_decision(cells)
    nv = U.norm_value(cells)
    rt = U.retest(cells)
    pe = U.position_effect(cells)
    final_txt = ", ".join(map(str, U.FINAL_LISTS))
    label = "최종" if stg["final"] else "잠정"
    now = datetime.now(U.KST)

    doc = Document()
    _style(doc)

    # ---------------- 표지 ----------------
    doc.add_heading("Quick-SIN 한국어판 실험 4 결과 보고서", level=0)
    _para(doc, "13개 리스트 최종 검증 · 검사-재검사 신뢰도 · 정상 기준", size=12, color="#555555")
    _para(doc, f"생성: {now:%Y-%m-%d %H:%M} (KST, 분석 페이지 자동 생성)", size=9.5)
    _para(doc, f"상태: {label} — 1차 {stg['n_r1']}명, 2차 {stg['n_r2']} / {stg['n_r1']}명 완료"
               + (f", 2차 미참여 {stg['n_drop']}명" if stg["n_drop"] else ""),
          bold=True, size=10.5, color=None if stg["final"] else C2)
    if not stg["final"]:
        _para(doc, "※ 2차(재검사)가 진행 중이라 검사-재검사 결과는 잠정이거나 비어 있습니다. "
                   "모든 참가자의 2차가 끝난 뒤 다시 내려받으면 최종 보고서가 됩니다.", size=9.5, color=C2)

    # ---------------- 1. 요약 ----------------
    doc.add_heading("1. 요약 및 결정 요청 사항", level=1)
    if rt["ready"]:
        ml, sg = rt["mean_level"], rt["single"]
        rel = (f"검사-재검사({rt['n']}명): 12개 평균 ICC {ml['icc']:.2f}, 리스트 1개 ICC 평균 {sg['icc_mean']:.2f}, "
               f"리스트 1개 CD95 {sg['cd95']:.1f} dB" + ("" if stg["final"] else " (잠정)"))
    else:
        rel = f"검사-재검사: 2차 진행 중({rt['n']}명 완료) — 2차 완료 후 산출"
    _bullets(doc, [
        f"최종 리스트 12개: {final_txt} (리스트 4 제외). 12개는 동등성 기준 3개를 모두 통과했다"
        f"(리스트 간 SD {e12['between_sd']:.2f} dB, 리스트 효과 p {p_fmt(e12['anova']['p_gg'])}).",
        f"정상 기준: {nv['value_rounded']:.1f} dB (95% CI {nv['ci'][0]:.1f} ~ {nv['ci'][1]:.1f} dB, {nv['n']}명). "
        "SNR 손실 = 환자 SNR-50 − 정상 기준.",
        rel,
        f"참가자: 정상 청력 성인 {stg['n_r1']}명(청력 기준 미달 {n_excluded}명 제외)." if n_excluded is not None
        else f"참가자: 정상 청력 성인 {stg['n_r1']}명.",
    ])
    doc.add_heading("결정 요청 사항", level=2)
    _table(doc, pd.DataFrame([
        ("① 최종 리스트", f"12개({final_txt}) 확정", "승인 — 사전 분석 계획대로 판정, 민감도 분석에서도 결론 동일"),
        ("② 정상 기준", f"{nv['value_rounded']:.1f} dB (A 규칙)", "승인 — 사전 규칙 적용, 12·13·11개 어느 세트로도 같은 값"),
        ("③ 난청군 후속 연구", "본 연구 전 난청 환자 예비 검사", "실시 — 현재 사다리로는 SNR 손실 약 9 dB까지만 측정 가능(6장)"),
    ], columns=["항목", "내용", "추천"]), widths=[3.2, 5.2, 8.2])

    # ---------------- 2. 배경 ----------------
    doc.add_heading("2. 연구 배경과 흐름", level=1)
    _para(doc, "QuickSIN 한국어판은 소음 속 문장 인지 능력을 짧게 측정하는 검사다. 문장 난이도 정규화(실험 1~2), "
               "리스트 구성과 동등성 확인(실험 3)을 거쳐, 실험 4에서 실험 3이 고른 13개 리스트를 새 참가자에게 "
               "최종 검증하고 검사-재검사 신뢰도와 정상 기준을 확인했다. 소음은 전 실험에서 4명 화자 babble로 같다.")
    _table(doc, HISTORY, widths=[2.0, 3.4, 5.6, 5.6], font=8.5)

    # ---------------- 3. 방법 ----------------
    doc.add_heading("3. 방법", level=1)
    doc.add_heading("3.1 참가자", level=2)
    _bullets(doc, [
        f"정상 청력 성인 {stg['n_r1']}명(실험 3 참가자 제외). Williams 26행 순서표를 한 번씩 채웠다.",
        "청력 기준: 양쪽 귀 250~8000 Hz 모든 주파수 역치 20 dB HL 이하."
        + (f" 미달자 {n_excluded}명은 분석에서 제외했다(데이터는 별도 보관)." if n_excluded else ""),
        "재검사(2차): 1차 후 1~2주(14일 이내), 1차와 같은 순서.",
    ])
    doc.add_heading("3.2 검사 자료와 절차", level=2)
    _bullets(doc, [
        "리스트: 실험 3에서 고른 13개(3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18). 리스트당 문장 6개, 문장당 키워드 5개.",
        "문장마다 SNR이 고정되어 있다: +4, +1, −2, −5, −8, −11 dB(3 dB 간격). 리스트 내부 구성은 실험 3과 같다.",
        "세션: 연습 리스트 2개(8 → 9) 후 본 검사 13개 리스트. 연습 리스트는 분석에 쓰지 않았다.",
        "제시 순서: 13개 리스트 Williams 설계 26행. 참가자 n번은 (n − 1) mod 26번째 행을 받는다. "
        "26명이 모이면 모든 리스트가 각 제시 순서와 앞뒤 짝에 같은 횟수로 나온다.",
    ])
    doc.add_heading("3.3 채점", level=2)
    _para(doc, "리스트별 정답 키워드 수(0~30)로 SNR-50(키워드의 절반을 알아듣는 신호대잡음비)을 계산했다. "
               "SNR-50 = 5.5 − 0.6 × 정답 키워드 수. 값이 낮을수록 잘 듣는 것이며, 키워드 1개는 0.6 dB다. "
               "예: 30개 중 15개를 맞히면 5.5 − 9.0 = −3.5 dB.")
    doc.add_heading("3.4 판정 기준 (수집 전 2026-11-18 확정한 분석 계획)", level=2)
    _bullets(doc, [
        f"리스트 동등성(1차 데이터): 리스트 간 SNR-50 SD < {U.CRIT_BETWEEN_LIST_SD} dB, 리스트 간 점수 SD < {U.CRIT_SCORE_SD}점, "
        f"반복측정 ANOVA 리스트 효과 p ≥ {U.CRIT_ANOVA_P}(Greenhouse-Geisser 보정). 미달 시 평균에서 가장 먼 리스트를 하나씩 "
        f"제외하며 다시 확인하고, 최종 리스트는 {U.MIN_LISTS}개 이상 유지한다.",
        "2차(재검사) 데이터는 같은 문장을 다시 듣는 검사라 리스트 구성에는 쓰지 않고 신뢰도 계산에만 쓴다.",
        "검사-재검사: ICC(2,1) 절대 일치(Koo & Li: .5 / .75 / .9 = 보통 / 좋음 / 매우 좋음), SEM = 회차 차이 SD / √2, "
        "CD95 = 1.96 × √2 × SEM, Bland-Altman, 대응표본 t. 최종 리스트 평균 수준과 리스트 1개 수준을 모두 산출한다.",
    ])
    _para(doc, "정상 기준 규칙", bold=True)
    _bullets(doc, [
        "A = 1차 제시 순서 1~2번째 리스트의 SNR-50 평균. 연습 2개 후 처음 듣는 리스트로, 환자가 임상에서 받는 조건과 같다.",
        "B = 1차 리스트 전체의 SNR-50 평균. 리스트를 많이 평균해 안정적이지만, 연달아 들으며 생기는 학습 효과가 섞인다.",
        f"|A − B| < {U.NORM_RULE_DIFF} dB이면 B, 그 이상이면 A를 쓴다. 0.5 dB는 키워드 1개(0.6 dB)보다 작은 차이로, "
        "임상적으로 의미 없는 수준이다.",
        "2차 데이터는 학습 효과가 섞이므로 쓰지 않는다. 최종값은 소수 첫째 자리로 반올림한다.",
    ])

    # ---------------- 4. 결과 ----------------
    doc.add_heading("4. 결과", level=1)
    doc.add_heading("4.1 수집 현황", level=2)
    iv = status["간격(일)"].dropna()
    d1 = status["1차 날짜"].dropna()
    _bullets(doc, [
        f"1차: {stg['n_r1']}명 완료 ({min(d1):%Y-%m-%d} ~ {max(d1):%Y-%m-%d}). 저장 누락·순서표 불일치 없음."
        if len(d1) else "1차: 데이터 없음",
        f"2차: {stg['n_r2']}명 완료" + (f", 간격 평균 {iv.mean():.1f}일(범위 {int(iv.min())}~{int(iv.max())}일)" if len(iv) else "")
        + (f", 대기 {stg['n_wait']}명" if stg["n_wait"] else "") + (f", 미참여 {stg['n_drop']}명" if stg["n_drop"] else "") + ".",
    ])

    doc.add_heading("4.2 리스트 동등성", level=2)
    steps = fd["steps1"]
    _para(doc, f"계획 원안 13개는 리스트 간 SD({e13['between_sd']:.2f} dB)와 점수 SD({e13['between_sd'] / .6:.2f}점) 기준은 "
               f"통과했지만, 리스트 효과가 유의했다(F({e13['anova']['df1']},{e13['anova']['df2']}) = {e13['anova']['F']:.2f}, "
               f"GG p {p_fmt(e13['anova']['p_gg'])}, 부분 η² = {e13['anova']['pes']:.2f}). 사전 규칙대로 평균에서 가장 먼 "
               f"리스트 4(평균보다 {steps[0].get('drop_dev', 0):+.2f} dB, 가장 쉬움)를 제외하자 12개에서 세 기준을 모두 "
               f"통과했다(리스트 간 SD {e12['between_sd']:.2f} dB, GG p {p_fmt(e12['anova']['p_gg'])}).")
    _table(doc, pd.DataFrame([{
        "단계": i + 1, "리스트 수": s["n_lists"], "리스트 간 SD (dB)": f"{s['between_sd']:.2f}",
        "점수 SD (점)": f"{s['between_sd'] / .6:.2f}", "리스트 효과 p": p_fmt(s["p_gg"]).replace("= ", ""),
        "판정": "통과" if s["passed"] else "미달", "제외": f"리스트 {s['drop']}" if "drop" in s else "—",
    } for i, s in enumerate(steps)]))
    _figure(doc, _fig_lists(e13, U.FINAL_LISTS),
            caption="그림 1. 리스트별 평균 SNR-50(1차). 점선은 13개 평균, 막대는 95% 신뢰구간.")
    ls = e13["lists"].copy()
    ls["판정"] = np.where(ls["list_no"].isin(U.FINAL_LISTS), "사용", "제외")
    ls12 = e12["lists"].set_index("list_no")
    _table(doc, pd.DataFrame({
        "리스트": ls["list_no"], "판정": ls["판정"], "평균 SNR-50 (dB)": ls["mean"].map("{:.2f}".format),
        "참가자 간 SD": ls["sd"].map("{:.2f}".format), "13개 평균과 차이": ls["dev"].map("{:+.2f}".format),
        "12개 평균과 차이": ls["list_no"].map(lambda l: f"{ls12.loc[l, 'dev']:+.2f}" if l in ls12.index else "—"),
    }))
    _para(doc, "리스트 4는 다른 리스트와 하나씩 비교했을 때는 유의하지 않아(Holm 보정 p = .14) 경계선 사례다. "
               "다음으로 평균에서 먼 리스트는 11(가장 어려움)이지만, 12개에서 이미 기준을 통과해 더 제외하지 않았다(5장).",
          size=9.5)

    doc.add_heading("4.3 제시 순서 효과", level=2)
    p1 = pe[pe["test_round"] == 1].set_index("presentation_order")["mean"]
    slope = np.polyfit(p1.index.astype(float), p1.values, 1)[0]
    _para(doc, f"1차에서 첫 번째 리스트는 평균 {p1.iloc[0]:.1f} dB, 마지막(13번째)은 {p1.iloc[-1]:.1f} dB로, 리스트를 하나 더 "
               f"들을 때마다 약 {abs(slope):.2f} dB씩 좋아졌다(학습 효과). Williams 설계로 각 리스트가 모든 순서에 같은 횟수로 "
               "나오므로 리스트 비교에는 영향이 없지만, 정상 기준을 정할 때는 이 효과를 고려했다(4.4).")
    _figure(doc, _fig_position(pe), caption="그림 2. 제시 순서별 평균 SNR-50.")

    doc.add_heading("4.4 정상 기준", level=2)
    _table(doc, pd.DataFrame([
        ("A", "1차 제시 순서 1~2번째 리스트 평균(최종 12개 중)", f"{nv['A']:.2f}", f"{nv['A_sd']:.2f}"),
        ("B", "1차 최종 12개 리스트 전체 평균", f"{nv['B']:.2f}", f"{nv['B_sd']:.2f}"),
    ], columns=["", "계산", "평균 (dB)", "참가자 간 SD (dB)"]), widths=[1.0, 9.0, 3.0, 3.5])
    _para(doc, f"|A − B| = {abs(nv['diff']):.2f} dB로 0.5 dB 이상이어서 규칙에 따라 {nv['use']}를 채택했다. "
               f"정상 기준은 {nv['value_rounded']:.1f} dB(계산값 {nv['value']:.2f} dB, 95% CI {nv['ci'][0]:.2f} ~ "
               f"{nv['ci'][1]:.2f} dB, {nv['n']}명)이다. 1·2번째에 리스트 4가 있었던 참가자는 나머지 1개 값을 썼다.")
    _bullets(doc, [
        f"A는 한 사람당 리스트 1~2개로 계산해 B보다 덜 정밀하다(참가자 간 SD {nv['A_sd']:.2f} vs {nv['B_sd']:.2f} dB). "
        "신뢰구간 폭이 약 ±0.5 dB이므로 '−3.3 dB 근처'로 이해한다." if nv["use"] == "A" else
        "B는 리스트를 많이 평균해 정밀하다.",
        "13개·11개 리스트로 계산해도 같은 값이다(5장).",
        f"참고(계획서 외): 정상 상한 = 정상 기준 + 1.645 × SD = {nv['upper']:.1f} dB. 정상인의 약 95%가 이보다 좋은 점수를 받는다. "
        "정상/경도 등 등급 기준은 난청군 후속 연구에서 정한다.",
        "이전 분석 페이지의 'K-QuickSIN 규준 −4.16 dB'는 실험 2-1-1의 문장 평균으로, 이 정상 기준과는 다른 값이다.",
    ])

    doc.add_heading("4.5 검사-재검사 신뢰도", level=2)
    if not rt["ready"]:
        _para(doc, f"2차(재검사) 진행 중이다({rt['n']}명 완료, 계산은 {U.MIN_N_RETEST}명 이상부터). "
                   "모든 참가자의 2차가 끝난 뒤 보고서를 다시 내려받으면 이 절이 채워진다.", color=C2)
    else:
        ml, sg = rt["mean_level"], rt["single"]
        if not stg["final"]:
            _para(doc, f"※ 잠정: 현재 {rt['n']}명 기준이다.", color=C2, size=9.5)
        _table(doc, pd.DataFrame([
            ("ICC(2,1) [95% CI]", f"{ml['icc']:.2f} [{ml['lo']:.2f}, {ml['hi']:.2f}] · {U.icc_label(ml['icc'])}",
             f"평균 {sg['icc_mean']:.2f} (범위 {sg['icc_min']:.2f}~{sg['icc_max']:.2f}) · {U.icc_label(sg['icc_mean'])}"),
            ("SEM", f"{ml['sem']:.2f} dB", f"{sg['sem']:.2f} dB"),
            ("CD95", f"{ml['cd95']:.2f} dB", f"{sg['cd95']:.2f} dB"),
            ("2차 − 1차 평균 차이", f"{ml['diff_mean']:+.2f} dB (t({ml['n'] - 1}) = {ml['t']:.2f}, p {p_fmt(ml['p'])})",
             f"{sg['diff_mean']:+.2f} dB"),
            ("95% 일치 한계", f"{ml['loa'][0]:+.2f} ~ {ml['loa'][1]:+.2f} dB", "—"),
        ], columns=["지표", "12개 리스트 평균", "리스트 1개"]), widths=[4.0, 6.5, 6.0])
        _para(doc, f"임상에서 리스트 1개로 검사하면, 같은 사람을 다시 검사했을 때 {sg['cd95']:.1f} dB보다 크게 달라져야 실제 변화로 "
                   "볼 수 있다(CD95). 2차 − 1차 차이가 음수이면 재검사에서 점수가 좋아진 것(학습 효과)이다.")
        _figure(doc, _fig_retest(rt), caption="그림 3. 1차 vs 2차(왼쪽)와 Bland-Altman(오른쪽, 주황 점선 = 95% 일치 한계).")
        pl = rt["per_list"]
        _table(doc, pd.DataFrame({
            "리스트": pl["list_no"], "1차 평균": pl["r1_mean"].map("{:.2f}".format), "2차 평균": pl["r2_mean"].map("{:.2f}".format),
            "차이 SD": pl["diff_sd"].map("{:.2f}".format), "ICC [95% CI]": pl.apply(lambda r: f"{r.icc:.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1),
            "표시": np.where(pl["unstable"], "들쭉날쭉", ""),
        }), font=8.5)
        _para(doc, f"'들쭉날쭉' = 회차 차이 SD가 리스트 중앙값({rt['stability_median']:.2f} dB)의 1.5배 초과. 표시만 하고 구성은 바꾸지 않았다.",
              size=9.5)

    # ---------------- 5. 계획 대비 ----------------
    doc.add_heading("5. 사전 계획과 달라진 점, 민감도 분석", level=1)
    _bullets(doc, [
        "계획 원안 13개가 리스트 효과 기준에 미달해, 사전 규칙대로 리스트 4를 제외하고 12개를 최종 리스트로 정했다.",
        "정상 기준과 신뢰도는 계획서의 '13리스트' 대신 최종 12개로 계산했다. 13개로 계산해도 정상 기준은 같다.",
        "참가자는 계획(30명 = 26행 + 예비 4명) 대신 26명(Williams 26행 1주기)으로 마쳤다.",
        "검사-재검사 결과는 리스트 구성에 쓰지 않았다(같은 문장을 다시 듣는 검사이므로).",
    ])
    sens = fd["sensitivity"].copy()
    sens["리스트 효과 p"] = sens["리스트 효과 p"].map(lambda p: p_fmt(p).replace("= ", ""))
    sens["세 기준"] = sens["세 기준"].str.replace("✅ ", "").str.replace("❌ ", "")
    _table(doc, sens, font=8.5)

    # ---------------- 6. 제한점 ----------------
    doc.add_heading("6. 제한점과 후속 연구 제안", level=1)
    _bullets(doc, [
        "정상 청력 성인에서만 검증한 결과다. 난청군에서의 리스트 동등성은 다시 확인해야 한다.",
        f"측정 범위: 현재 사다리(+4 ~ −11 dB)로 잴 수 있는 가장 나쁜 SNR-50은 +5.5 dB(정답 0개)다. 정상 기준이 "
        f"{nv['value_rounded']:.1f} dB이므로 측정 가능한 SNR 손실은 최대 약 {5.5 - nv['value_rounded']:.0f} dB다. "
        "중등도 이상 난청에서는 바닥 효과로 구분이 어려울 수 있다.",
        "사다리 범위를 넓히면 문장–SNR 짝이 바뀌므로 리스트 동등성을 다시 확인해야 한다. 본 연구 전에 난청 환자 몇 명으로 "
        "예비 검사를 해 바닥 효과 정도를 확인할 것을 제안한다.",
        "정상 기준 A는 1인당 리스트 1~2개로 계산해 신뢰구간이 약 ±0.5 dB다.",
    ])

    # ---------------- 부록 ----------------
    doc.add_heading("부록 A. 참가자별 1차 점수 (SNR-50, dB)", level=1)
    a = m13.copy().round(1)
    a.insert(0, "참가자", a.index)
    a.columns = ["참가자"] + [f"L{c}" for c in m13.columns]
    _table(doc, a, font=7)
    doc.add_heading("부록 B. 제시 순서표 (Williams 26행)", level=1)
    w = pd.DataFrame(U.WILLIAMS26, columns=[f"{i}번째" for i in range(1, 14)])
    w.insert(0, "참가자", [f"{i + 1}" for i in range(26)])
    _table(doc, w, font=7)

    _para(doc, "이 보고서는 Quick-SIN 분석 페이지(실험 4)에서 자동 생성되었다. 계산 코드: modules/exp4_utils.py, "
               "modules/exp4_report.py.", size=8.5, color="#777777", italic=True)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
