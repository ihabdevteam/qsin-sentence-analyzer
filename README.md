# qsin-sentence-analyzer

Quick-SIN 한국어판 개발 실험 데이터를 분석하는 Streamlit 앱입니다. 결과는 DB 데이터로 매번 다시 계산합니다.

## 페이지

| 페이지 | 파일 | 내용 |
|---|---|---|
| Exp3 | `pages/1_Exp3.py` | 실험 3: 18개 리스트 → 최종 13개 선정 근거, 데이터 ZIP |
| Exp4 | `pages/2_Exp4.py` | 실험 4: 수집 현황, 데이터 점검, 리스트 동등성(최종 12개), 순서 효과, 정상 기준, 검사-재검사, 데이터 ZIP, Word 보고서 |

`legacy/`에는 예전 페이지(더미 데이터 생성기, 실험 1~2용 문장 분석기, 피험자 선택 분석)가 보관되어 있습니다.
`pages/` 밖이라 앱 메뉴에 나오지 않습니다. 생성기는 DB에 테스트 데이터를 쓰므로 다시 메뉴에 올리지 마세요.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `modules/db_utils.py` | DB 연결 |
| `modules/exp3_utils.py` | 실험 3 데이터 로더 |
| `modules/exp3_selection.py` | 실험 3 리스트 판정(기준 A~E, 단계적 제외, 민감도). 기준값·확정 결과 상수 고정 |
| `modules/exp4_utils.py` | 실험 4 로더, 순서표, 수집 현황, 데이터 점검, 동등성, 정상 기준, 검사-재검사, 최종 리스트. 기준값·확정 결과 상수 고정 |
| `modules/exp4_report.py` | 실험 4 Word 보고서 생성 |

판정 기준과 확정 결과(`FINAL_LISTS`)는 수집 전 분석 계획에 따라 코드에 고정되어 있습니다. 화면에서 바꿀 수 없고,
DB로 다시 계산한 결과가 확정 결과와 다르면 페이지에 경고가 뜹니다. 기준을 바꿔야 한다면 연구 책임자와 상의한 뒤
코드와 판정 문서를 함께 고치세요.

## 데이터 규칙

- 실험 구분은 리포트의 `experiment` 값(`exp3`, `exp4`)으로 합니다. 분석에서 빼야 하는 참가자는 데이터를 지우지 않고
  `experiment`를 `exp4_excluded`처럼 바꿉니다.
- 실험 4: `test_round`(1 = 1차, 2 = 재검사), `is_practice`(연습 리스트 8·9).
- SNR-50 = 5.5 − 0.6 × 리스트 정답 키워드 수(사다리 +4 ~ −11 dB, 3 dB 간격, 문장 6개 × 키워드 5개).
- 다운로드 파일과 보고서에는 이름 등 개인정보를 넣지 않고 참가자 번호만 씁니다. 검사 문장 원문도 넣지 않습니다.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

`.streamlit/secrets.toml`에 `SUPABASE_URL`, `SUPABASE_KEY`가 필요합니다(저장소에 올리지 않음).
배포(Streamlit Community Cloud)에서는 앱 설정의 Secrets에 같은 값을 넣습니다. Secrets에 `APP_PASSWORD = "..."`를 추가하면 모든 페이지가
비밀번호를 묻습니다(없으면 공개, `modules/auth.py`). `packages.txt`의 `fonts-nanum`은
보고서 그래프의 한글 글꼴용입니다.

## 작업 흐름

이슈 → `feat/#번호-설명` 브랜치 → `staging` PR → `staging` → `main` PR(배포).
