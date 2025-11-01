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
default_database_url = os.getenv("DATABASE_URL", "sqlite:////data/sqlite/owlv2_examples.db")

with st.sidebar:
    st.header("설정")

    # 서버 URL 설정
    server_url = st.text_input(
        "백엔드 서버 URL",
        value=default_backend_url,
        help="백엔드 API 서버의 URL을 입력하세요"
    )

    database_url_input = st.text_input(
        "데이터베이스 URL",
        value=default_database_url,
        help="OWLv2 예시 이미지 정보를 저장할 SQLite 데이터베이스 URL을 입력하세요"
    )
    normalized_db_url = database_url_input.strip()
    effective_database_url = (
        normalized_db_url
        if normalized_db_url and normalized_db_url != default_database_url
        else None
    )
    st.session_state["database_url_input"] = database_url_input
    st.session_state["database_url_effective"] = effective_database_url

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

    model_options = {
        "GroundingDINO": "grounding_dino",
        "OmDet Turbo (실험적)": "omdet_turbo",
        "OWLv2 (원샷)": "owlv2",
    }
    model_display_names = list(model_options.keys())
    default_index = model_display_names.index("GroundingDINO")
    selected_model_label = st.selectbox(
        "모델 선택",
        options=model_display_names,
        index=default_index,
        help="사용할 탐지 모델을 선택하세요. OmDet Turbo는 현재 실험적 상태입니다.",
    )
    selected_model = model_options[selected_model_label]

st.session_state.setdefault("owlv2_examples", [])
st.session_state.setdefault("owlv2_examples_error", None)
st.session_state.setdefault("owlv2_last_query", None)
st.session_state.setdefault("owlv2_last_database_url", None)
st.session_state.setdefault("owlv2_last_server_url", None)
st.session_state.setdefault("owlv2_last_model", None)
st.session_state.setdefault("database_url_input", default_database_url)
st.session_state.setdefault("database_url_effective", None)

if selected_model != "owlv2":
    st.session_state["owlv2_examples"] = []
    st.session_state["owlv2_examples_error"] = None
    st.session_state["owlv2_last_query"] = None
    st.session_state["owlv2_last_model"] = selected_model

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

    if selected_model == "owlv2":
        normalized_query = query.strip()
        db_url_for_examples = st.session_state.get("database_url_effective", effective_database_url)
        last_query = st.session_state.get("owlv2_last_query")
        last_db = st.session_state.get("owlv2_last_database_url")
        last_server = st.session_state.get("owlv2_last_server_url")
        last_model = st.session_state.get("owlv2_last_model")

        should_fetch_examples = (
            bool(normalized_query)
            and (
                normalized_query != last_query
                or db_url_for_examples != last_db
                or server_url != last_server
                or last_model != "owlv2"
            )
        )

        if should_fetch_examples:
            try:
                params = {"query": normalized_query}
                if db_url_for_examples:
                    params["database_url"] = db_url_for_examples
                examples_endpoint = server_url.rstrip("/") + "/owlv2/examples"
                response = requests.get(examples_endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                st.session_state["owlv2_examples"] = data.get("examples", [])
                st.session_state["owlv2_examples_error"] = None
            except requests.exceptions.RequestException as exc:
                st.session_state["owlv2_examples"] = []
                st.session_state["owlv2_examples_error"] = f"예시 이미지를 불러오지 못했습니다: {exc}"
            except ValueError:
                st.session_state["owlv2_examples"] = []
                st.session_state["owlv2_examples_error"] = "예시 이미지 응답을 해석할 수 없습니다."
            st.session_state["owlv2_last_query"] = normalized_query
            st.session_state["owlv2_last_database_url"] = db_url_for_examples
            st.session_state["owlv2_last_server_url"] = server_url
        elif not normalized_query:
            st.session_state["owlv2_examples"] = []
            st.session_state["owlv2_examples_error"] = None
            st.session_state["owlv2_last_query"] = None

        st.session_state["owlv2_last_model"] = "owlv2"

        st.markdown("#### OWLv2 예시 이미지")
        error_message = st.session_state.get("owlv2_examples_error")
        examples = st.session_state.get("owlv2_examples", [])

        if error_message:
            st.error(error_message)
        elif not normalized_query:
            st.info("검색어를 입력하면 등록된 예시 이미지를 확인할 수 있습니다.")
        elif not examples:
            st.info("등록된 예시 이미지가 없습니다.")
        else:
            for example in examples:
                caption = example.get("filename") or f"예시 이미지 #{example.get('id')}"
                image_data = example.get("image_base64")
                if not image_data:
                    st.warning(f"이미지 데이터가 없어 '{caption}'를 표시할 수 없습니다.")
                    continue
                try:
                    image_bytes = base64.b64decode(image_data)
                    st.image(image_bytes, caption=caption, use_column_width=True)
                except Exception:
                    st.warning(f"예시 이미지를 표시할 수 없습니다: {caption}")

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
                "model": selected_model,
            }
            payload_db_url = st.session_state.get("database_url_effective", effective_database_url)
            if payload_db_url:
                payload["database_url"] = payload_db_url

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
