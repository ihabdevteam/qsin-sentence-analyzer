import streamlit as st
import json
import random
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
        session_idx_no = st.text_input("세션 인덱스 번호", value="1")
    with c4:
        test_result = st.number_input("테스트 결과 (종합)", value=0.0, format="%.2f")

    st.markdown("##### 테스트 환경 설정")
    c5, c6, c7 = st.columns(3)
    with c5:
        receiver = st.selectbox("Receiver", ["Headphone", "Speaker"])
        fixed_type = st.text_input("Fixed Type", value="SF", disabled=True)
    with c6:
        direction = st.text_input("Direction", value="LR", disabled=True)
        volume_level = st.number_input("볼륨 레벨", value=0)
    with c7:
        # 수정된 부분: SNR 레벨을 selectbox로 변경
        snr_level = st.selectbox("SNR 레벨 (dB)", options=[-10, -5, 0, 5, 10, 15, 20, 25])
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
        with st.spinner("자동으로 360개 점수 데이터를 생성하고 데이터베이스에 전송 중입니다..."):
            
            # --- 수정된 부분: SNR 레벨에 따른 가중치 부여 함수 ---
            def generate_biased_score(snr):
                """
                SNR 레벨에 따라 가중치를 적용하여 점수를 생성합니다.
                SNR이 높을수록 높은 점수가 나올 확률이 높아집니다.
                """
                possible_scores = [0, 0.5, 1]
                
                if snr <= -10:
                    weights = [0.80, 0.15, 0.05]  # P(0), P(0.5), P(1)
                elif snr == -5:
                    weights = [0.60, 0.30, 0.10]
                elif snr == 0:
                    weights = [0.30, 0.40, 0.30]
                elif snr == 5:
                    weights = [0.10, 0.40, 0.50]
                elif snr == 10:
                    weights = [0.05, 0.25, 0.70]
                elif snr == 15:
                    weights = [0.05, 0.10, 0.85]
                else: # snr >= 20
                    weights = [0.02, 0.03, 0.95]
                
                return random.choices(possible_scores, weights=weights, k=1)[0]
            # --- 수정 종료 ---

            scores_payload = []
            for sentence_info in sentences_data:
                num_keywords = len(sentence_info.get("keyword", []))
                # 가중치 적용 함수 호출
                random_scores = [generate_biased_score(snr_level) for _ in range(num_keywords)]
                
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
                "p_volume_level": int(volume_level), "p_snr_level": int(snr_level),
                "p_memo": memo, "p_sound_set": int(sound_set),
                "p_test_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "p_test_result": float(test_result), "p_reg_timestamp": int(time.time()),
                "p_session_id": session_id, "p_session_idx_no": str(session_idx_no),
                "p_scores": scores_payload
            }
            try:
                data, error = supabase.rpc('create_qsin_report_with_scores', report_payload).execute()
                api_response = data[1] if data and len(data) > 1 else None
                if api_response:
                    st.success(f"성공적으로 테스트 리포트를 생성했습니다! (ID: {api_response})")
                    st.balloons()
                else:
                    st.error(f"데이터 생성 중 오류 발생: {error.message if error else '알 수 없는 오류'}")
            except Exception as e:
                st.error(f"RPC 호출 중 예외 발생: {e}")
