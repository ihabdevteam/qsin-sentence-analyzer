import streamlit as st
import json
import random
import uuid
import time
from datetime import datetime
from modules.db_utils import init_supabase_client, get_profiles, get_patients

st.set_page_config(page_title="리포트 생성기", layout="wide")
st.title("Quick-SIN 테스트 리포트 생성기 📝")
st.write("담당자와 환자를 선택하고 리포트 정보를 입력하면, 360개의 `score_qsin` 데이터를 자동으로 생성하여 데이터베이스에 삽입합니다.")

# --- 클라이언트 및 데이터 로딩 ---
supabase = init_supabase_client()
if not supabase:
    st.stop()

@st.cache_data
def load_sentences(file_path="sentences_data.json"):
    """sentences_data.json 파일을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"`{file_path}` 파일을 찾을 수 없습니다. `app.py`와 같은 디렉토리에 있는지 확인하세요.")
        return None

profiles_data = get_profiles(supabase)
patients_data = get_patients(supabase)
sentences_data = load_sentences()

# 로딩 상태 확인
loading_successful = all([profiles_data, patients_data, sentences_data])
with st.expander("데이터 로딩 상태 확인", expanded=not loading_successful):
    if profiles_data: st.success(f"✅ 담당자 목록 로드 성공: {len(profiles_data)}명")
    else: st.error("❌ 담당자 목록 로드 실패.")
    if patients_data: st.success(f"✅ 환자 목록 로드 성공: {len(patients_data)}명")
    else: st.error("❌ 환자 목록 로드 실패.")
    if sentences_data: st.success("✅ 문장 데이터 파일 로드 성공.")
    else: st.error("❌ 문장 데이터 파일 로드 실패.")

if not loading_successful:
    st.error("필수 데이터 로딩에 실패하여 앱을 중지합니다.")
    st.stop()

# --- 사용자 입력 폼 ---
profile_map = {p['tester_name']: p['user_id'] for p in profiles_data}
patient_map = {p['name']: p['id'] for p in patients_data}

with st.form(key="report_form"):
    st.markdown("##### 담당자 및 환자 선택")
    c1, c2 = st.columns(2)
    with c1:
        selected_tester_name = st.selectbox("담당자 선택", options=list(profile_map.keys()))
    with c2:
        selected_patient_name = st.selectbox("환자 선택", options=list(patient_map.keys()))

    st.markdown("##### 세션 및 테스트 정보")
    c3, c4 = st.columns(2)
    with c3:
        session_id = st.text_input("세션 ID", value=f"dummy_{int(time.time())}")    

    st.markdown("##### 테스트 환경 설정")
    c5, c6, c7 = st.columns(3)
    with c5:
        receiver = st.selectbox("Receiver", ["Headphone", "Speaker"])
        fixed_type = st.text_input("Fixed Type", value="SF", disabled=True)
    with c6:
        direction = st.text_input("Direction", value="LR", disabled=True)
        volume_level = st.number_input("볼륨 레벨", value=0)
    with c7:
        # SNR 레벨 선택은 비활성화, 대신 예측 평균 dB를 입력받음
        predicted_mean_db = st.number_input("예측 평균 dB", value=2.0, format="%.2f", help="로지스틱 50% 교차점을 설정합니다.")
        sound_set = st.number_input("사운드 세트 번호", value=0)

    memo = st.text_area("메모")
    submit_button = st.form_submit_button(label="🚀 리포트 생성 및 데이터베이스에 저장")

# --- 폼 제출 로직 ---
if submit_button:
    user_id = profile_map.get(selected_tester_name)
    patient_user_id = patient_map.get(selected_patient_name)
    if not all([user_id, patient_user_id]):
        st.warning("담당자와 환자를 선택해주세요.")
    else:        
        session_id_to_use = session_id       

        with st.spinner("SNR 그리드(-5, 0, 5)로 1080개 점수 데이터를 생성하여 데이터베이스에 전송 중입니다..."):
            SNR_GRID = [-5, 0, 5]

            def logistic_p(snr: float, center_db: float, slope_pct_per_db: float = 10.0) -> float:
                """주어진 SNR에서 로지스틱 함수 기반 정답 확률을 계산한다."""
                # slope(%/dB) -> coef b, intercept a = -b*center
                b = (slope_pct_per_db / 100.0) * 4.0
                a = -b * center_db
                z = a + b * snr
                return 1.0 / (1.0 + (2.718281828459045 ** (-z)))

            def sample_score_from_p(p: float, h_base: float = 0.1):
                """정답 확률 p로부터 가중 랜덤 점수(0, 0.5, 1)를 샘플링한다."""
                # 분포: w1=(1-h)p, w05=2hp, w0=1-w1-w05; h는 안정성을 위해 min으로 캡핑
                p = max(0.0, min(1.0, p))
                if p == 0.0:
                    return 0.0
                if p == 1.0:
                    return 1.0
                # h는 (1-p)/p 이하로 제한하여 w0>=0 보장
                h_max = (1.0 - p) / p
                h = min(h_base, h_max) if h_max > 0 else 0.0
                w1 = (1.0 - h) * p
                w05 = 2.0 * h * p
                w0 = max(0.0, 1.0 - w1 - w05)
                weights = [w0, w05, w1]
                return random.choices([0.0, 0.5, 1.0], weights=weights, k=1)[0]

            created_ids = []
            total_records = 0
            for idx, snr in enumerate(SNR_GRID, start=1):
                scores_payload = []
                for sentence_info in sentences_data:
                    num_keywords = len(sentence_info.get("keyword", []))
                    p = logistic_p(snr, center_db=predicted_mean_db, slope_pct_per_db=10.0)
                    random_scores = [sample_score_from_p(p) for _ in range(num_keywords)]
                    scores_payload.append({
                        "index": sentence_info["index"],
                        "sentences": sentence_info["sentences"],
                        "full_sentence": sentence_info["fullSentence"],
                        "score": random_scores,
                        "total_score": sum(random_scores)
                    })

                report_payload = {
                    "p_user_id": user_id, "p_patient_user_id": patient_user_id,
                    "p_receiver": receiver, "p_fixed_type": fixed_type, "p_direction": direction,
                    "p_volume_level": int(volume_level), "p_snr_level": int(snr),
                    "p_memo": memo, "p_sound_set": int(sound_set),
                    "p_test_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "p_test_result": 0.0, "p_reg_timestamp": int(time.time()),
                    "p_session_id": session_id_to_use, "p_session_idx_no": str(idx),
                    "p_scores": scores_payload
                }
                try:
                    data, error = supabase.rpc('create_qsin_report_with_scores', report_payload).execute()
                    api_response = data[1] if data and len(data) > 1 else None
                    if api_response:
                        created_ids.append(api_response)
                        total_records += len(scores_payload)
                    else:
                        st.error(f"SNR {snr} 처리 중 오류: {error.message if error else '알 수 없는 오류'}")
                except Exception as e:
                    st.error(f"RPC 호출 중 예외 발생 (SNR {snr}): {e}")

            if created_ids:
                st.success(f"성공적으로 {len(SNR_GRID)}개의 리포트를 생성했고, 총 {total_records}개 문장 점수를 삽입했습니다.")
                st.write(f"생성된 리포트 IDs: {created_ids}")
                st.info(f"사용된 세션 ID: {session_id_to_use}")
                st.balloons()
            else:
                st.error("어떤 리포트도 생성되지 않았습니다. 로그를 확인해주세요.")
