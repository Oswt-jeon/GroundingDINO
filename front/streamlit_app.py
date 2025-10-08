import base64
import os

import requests
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="GroundingDINO Search",
    page_icon="🔍",
    layout="wide"
)

# 제목
st.title("🔍 GroundingDINO Object Detection")
st.markdown("---")

# 사이드바에 설정 옵션
default_backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

with st.sidebar:
    st.header("설정")

    # 서버 URL 설정
    server_url = st.text_input(
        "백엔드 서버 URL",
        value=default_backend_url,
        help="백엔드 API 서버의 URL을 입력하세요"
    )

    # 임계값 설정
    box_threshold = st.slider(
        "Box Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05
    )

    text_threshold = st.slider(
        "Text Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    limit = st.number_input(
        "최대 결과 수",
        min_value=1,
        max_value=50,
        value=6,
        step=1,
        help="검색 결과로 반환될 최대 이미지 수"
    )

# 메인 컨텐츠
col1, col2 = st.columns([1, 1])

with col1:
    st.header("검색 입력")

    # 검색 입력
    query = st.text_input(
        "검색할 객체",
        placeholder="예: person . car . dog",
        help="찾고 싶은 객체들을 입력하세요. 여러 객체는 점(.)으로 구분합니다."
    )

    # 검색 버튼
    search_button = st.button(
        "🔍 검색하기",
        type="primary",
        disabled=not query,
        use_container_width=True
    )

with col2:
    st.header("검색 결과")
    
    # 결과 표시 영역
    result_container = st.empty()

# 검색 실행
if search_button and query:
    st.session_state["search_results"] = []
    with st.spinner("검색 중입니다..."):
        try:
            payload = {
                "text": query,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "limit": int(limit),
            }

            endpoint = server_url.rstrip("/") + "/search"
            response = requests.post(
                endpoint,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                st.session_state["search_results"] = results

                with result_container.container():
                    if not results:
                        st.warning("조건에 맞는 객체를 찾지 못했습니다.")
                    else:
                        st.success(f"검색이 완료되었습니다! ({len(results)}개 이미지)")
                        for i, item in enumerate(results, start=1):
                            st.markdown(f"### 결과 {i}: `{os.path.basename(item.get('image', ''))}`")

                            annotated = item.get("annotated_image")
                            if annotated and annotated.get("data"):
                                try:
                                    image_bytes = base64.b64decode(annotated["data"])
                                    st.image(image_bytes, caption="검출 결과", use_column_width=True)
                                except Exception:
                                    st.warning("주석 이미지를 표시할 수 없습니다.")
                            st.markdown("---")

            else:
                with result_container.container():
                    st.error(f"서버 오류: {response.status_code}")
                    st.text(response.text)

        except requests.exceptions.ConnectionError:
            with result_container.container():
                st.error("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        except requests.exceptions.Timeout:
            with result_container.container():
                st.error("요청 시간이 초과되었습니다.")
        except Exception as e:
            with result_container.container():
                st.error(f"오류가 발생했습니다: {str(e)}")

# 사용법 안내
with st.expander("📖 사용법"):
    st.markdown("""
    ### 사용 방법
    1. **검색할 객체 입력**: 찾고 싶은 객체 이름을 입력하세요
       - 여러 객체를 찾으려면 점(.)으로 구분하세요
       - 예: `person . car . dog`
    
    2. **임계값 조정**: 사이드바에서 검출 민감도를 조정할 수 있습니다
       - Box Threshold: 객체 검출 임계값
       - Text Threshold: 텍스트 매칭 임계값
    
    3. **검색 실행**: 모든 설정이 완료되면 검색 버튼을 클릭하세요
    
    ### 백엔드 서버 요구사항
    - FastAPI 백엔드(`inference` 서비스)가 실행 중이어야 합니다
    - `/search` 엔드포인트가 활성화되어 있어야 합니다
    """)

# 실시간 상태 표시
st.markdown("---")
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if query:
        st.success("✅ 검색어 입력됨")
    else:
        st.warning("⚠️ 검색어를 입력하세요")

with status_col2:
    result_count = len(st.session_state.get("search_results", []))
    if result_count:
        st.success(f"✅ {result_count}개 결과")
    else:
        st.info("ℹ️ 결과 없음")

with status_col3:
    try:
        # 서버 상태 확인
        health_endpoint = server_url.rstrip("/") + "/healthz"
        response = requests.get(health_endpoint, timeout=2)
        if response.status_code == 200:
            st.success("✅ 서버 연결됨")
        else:
            st.error("❌ 서버 오류")
    except:
        st.error("❌ 서버 연결 실패")
