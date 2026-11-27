"""간단한 비밀번호 확인.

Streamlit Secrets에 APP_PASSWORD가 있으면 페이지를 열 때 비밀번호를 묻는다. 없으면 그대로 연다(공개).
관리자가 앱 설정 → Secrets에 아래 한 줄을 추가하면 코드를 고치지 않아도 바로 적용된다.

    APP_PASSWORD = "원하는비밀번호"
"""

import hmac

import streamlit as st


def require_password() -> None:
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        pw = None
    if not pw or st.session_state.get("auth_ok"):
        return
    st.title("Quick-SIN 분석 페이지")
    entered = st.text_input("비밀번호", type="password")
    if entered:
        if hmac.compare_digest(entered, str(pw)):
            st.session_state.auth_ok = True
            st.rerun()
        st.error("비밀번호가 맞지 않습니다.")
    st.stop()
