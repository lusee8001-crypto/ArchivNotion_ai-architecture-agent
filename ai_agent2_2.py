import streamlit as st
import requests
import os
import base64
import io
import re
import yaml
import secrets
from pathlib import Path
from google import genai
from google.genai import types

# ─── 선택적 의존성 임포트 ─────────────────────────────────────────────────────

try:
    import streamlit_authenticator as stauth
except ImportError:
    stauth = None

try:
    import bcrypt
except ImportError:
    bcrypt = None


# ════════════════════════════════════════════════════════════════════════════════
# 1. 환경변수 & API 클라이언트
# ════════════════════════════════════════════════════════════════════════════════

NOTION_TOKEN          = os.getenv("NOTION_TOKEN",           "your_notion_token_here")
NOTION_MAIN_DB_ID     = os.getenv("NOTION_MAIN_DB_ID",      "your_notion_main_db_id_here")
NOTION_RELATION_PROP  = os.getenv("NOTION_RELATION_PROP_NAME", "프로젝트")
# 새 프로젝트 폴더를 생성할 상위(부모) 데이터베이스 ID
# 실제 배포 시 환경변수 NOTION_PROJECTS_DB_ID 에 노션 프로젝트 DB ID를 주입하세요.
NOTION_PROJECTS_DB_ID = os.getenv("NOTION_PROJECTS_DB_ID",  "")

# 기본 프로젝트 맵 (환경변수로 재정의 가능)
_DEFAULT_PROJECT_MAP: dict[str, str] = {
    "[설계 스튜디오] 신사동 복합시설": os.getenv("NOTION_PAGE_ID_PROJECT1", ""),
    "[건축이론] 현대건축 사조 리서치": os.getenv("NOTION_PAGE_ID_PROJECT2", ""),
    "기타 개인 리서치":               os.getenv("NOTION_PAGE_ID_PROJECT3", ""),
}

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
client_gemini   = genai.Client(api_key=GEMINI_API_KEY)


# ════════════════════════════════════════════════════════════════════════════════
# 2. 인증 설정 — config.yaml 자동 초기화
# ════════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path("config.yaml")

def _init_auth_config() -> None:
    """config.yaml이 없으면 기본 계정(admin / admin123)으로 최초 생성."""
    if CONFIG_PATH.exists() or bcrypt is None:
        return
    pw_hash = bcrypt.hashpw(b"admin123", bcrypt.gensalt(12)).decode()
    initial: dict = {
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "admin@archivnotion.com",
                    "name": "ArchivAdmin",
                    "password": pw_hash,
                }
            }
        },
        "cookie": {
            "expiry_days": 30,
            "key": secrets.token_hex(32),
            "name": "archivnotion_signature",
        },
        "preauthorized": {"emails": []},
    }
    CONFIG_PATH.write_text(
        yaml.dump(initial, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


def _save_auth_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


_init_auth_config()

with open(CONFIG_PATH, encoding="utf-8") as _f:
    auth_config: dict = yaml.load(_f, Loader=yaml.SafeLoader)


# ════════════════════════════════════════════════════════════════════════════════
# 3. Notion API 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def _notion_headers() -> dict:
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }


def create_notion_project(folder_name: str) -> tuple[str | None, str | None]:
    """
    노션 프로젝트 DB(NOTION_PROJECTS_DB_ID)에 신규 페이지 생성.
    성공 시 (page_id, page_url), 실패 시 (None, error_message) 반환.
    """
    if not NOTION_PROJECTS_DB_ID:
        return None, "NOTION_PROJECTS_DB_ID 환경변수가 설정되지 않았습니다."

    payload = {
        "parent": {"database_id": NOTION_PROJECTS_DB_ID},
        "properties": {
            "이름": {"title": [{"text": {"content": folder_name}}]},
        },
    }
    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=_notion_headers(),
        json=payload,
    )
    if res.status_code == 200:
        data = res.json()
        return data.get("id", ""), data.get("url", "")
    return None, f"API 오류 {res.status_code}: {res.text}"


def _relation_prop(project_folder: str) -> dict | None:
    """선택된 프로젝트의 페이지 ID → Notion relation 속성 딕셔너리."""
    project_map: dict = st.session_state.get("project_map", _DEFAULT_PROJECT_MAP)
    page_id = project_map.get(project_folder, "").replace("-", "").strip()
    return {"relation": [{"id": page_id}]} if page_id else None


# ════════════════════════════════════════════════════════════════════════════════
# 4. Gemini 멀티모달 분석 엔진
# ════════════════════════════════════════════════════════════════════════════════

def _image_mime(filename: str) -> str:
    return "image/jpeg" if filename.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg") else "image/png"


def analyze_gemini(
    system_prompt: str,
    source_url: str,
    file_bytes: bytes | None,
    source_type: str,
    file_name: str = "",
) -> str:
    """
    Gemini 2.5 Flash 멀티모달 호출.
    - URL   : 텍스트 지시
    - 이미지 : Part.from_bytes(image/*)
    - PDF   : Part.from_bytes(application/pdf) — 네이티브 레이아웃 분석
    """
    base_instruction = (
        "제시된 지침과 출력 포맷을 100% 준수하여 아카이빙용 크리틱 리포트를 생성해 주세요."
    )
    contents: list = []

    if source_type == "URL 링크":
        contents.append(f"{base_instruction}\n\n분석 대상 URL: {source_url}")
    elif source_type == "사진/이미지 파일" and file_bytes:
        mime = _image_mime(file_name)
        contents.extend([
            types.Part.from_bytes(data=file_bytes, mime_type=mime),
            base_instruction,
        ])
    elif source_type == "PDF 문서" and file_bytes:
        contents.extend([
            types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
            base_instruction,
        ])

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=8192,
            temperature=0.2,
        ),
    )
    return response.text


# ════════════════════════════════════════════════════════════════════════════════
# 5. 시스템 프롬프트 빌더 (기능 1 · 2 · 4 통합)
# ════════════════════════════════════════════════════════════════════════════════

def build_system_prompt(layers_text: str, has_custom_layer: bool) -> str:
    custom_layer_rule = ""
    if has_custom_layer:
        custom_layer_rule = """
[커스텀 크리틱 레이어 처리 규칙]
- layers_text 안에 '🎯 [사용자 커스텀 레이어]: (내용)' 항목이 있다면,
  반드시 아래 섹션 구분자를 사용하여 별도 섹션으로 분석하세요.
  ===SECTION: 🎯 사용자 커스텀 레이어===
"""

    return f"""
당신은 전세계 건축 논문, 도면, 사진을 분석하는 세계 최고 수준의 AI 건축 크리틱 파트너입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[기능 1 — 출력 스타일 규칙 · 절대 위반 금지]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 서술형 줄글 작성 절대 금지. 모든 분석은 넘버링 + 개조식(Bullet) 혼합 형식으로 작성.
② 기본 형식:
   1. **핵심 항목명** — 세부 추론 내용을 간결하게 서술.
      ↳ 보조 설명이 필요하면 들여쓰기( ↳)를 활용한 하위 bullet 추가.
③ 건축 키워드·개념어·고유명칭은 반드시 **볼드**(**) 처리.
④ 각 섹션 내 항목은 최소 3개 ~ 최대 7개로 정리.
⑤ 모든 본문은 존댓말(~합니다/~됩니다/~보입니다)로 작성.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[기능 2 — 영문 자료 번역 & 원문 병기 섹션 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 입력 자료(PDF·이미지·URL)에 영어 텍스트가 1문장이라도 포함된 경우,
  리포트 전체의 최상단 첫 번째 섹션으로 반드시 아래 섹션을 생성하세요.

===SECTION: 🌐 영문 자료 번역 및 원문 대조===
(아래 형식을 단락·문장 단위로 반복)

1. **[단락/문장 요약 제목]**
   - [번역]: (한국어 건축 전문 용어로 매끄럽게 번역된 내용)
   - [원문]: (이에 대응하는 Original English Text)

2. **[다음 단락 요약 제목]**
   - [번역]: ...
   - [원문]: ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 기본 지침]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 유명세나 건축가 이름에 의존하지 말고, 물리적·조형적 단서를 바탕으로
  왜 이런 설계 디테일이 도출되었는지 깊이 있는 '추론성 분석'을 작성하세요.
{custom_layer_rule}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[집중 분석 레이어]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{layers_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 포맷 — 파싱을 위해 100% 엄격히 준수]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
각 분석 섹션을 반드시 아래 구분자 형식으로 감싸 출력하세요.
섹션 제목은 위 레이어명 문자열을 한 글자도 변경 없이 그대로 사용하세요.

===SECTION: 섹션제목===
(넘버링 + 개조식 분석 내용)
===SECTION: 다음섹션제목===
...
    """.strip()


# ════════════════════════════════════════════════════════════════════════════════
# 6. Notion 블록 파싱 & 생성 파이프라인
# ════════════════════════════════════════════════════════════════════════════════

_SECTION_RE = re.compile(
    r"===SECTION:\s*(.+?)===[ \t]*\n([\s\S]+?)(?=\n?===SECTION:|\Z)",
    re.MULTILINE,
)


def _parse_sections(text: str) -> list[tuple[str, str]]:
    matches = _SECTION_RE.findall(text)
    if matches:
        return [(t.strip(), c.strip()) for t, c in matches]

    parts = re.split(r"\n##\s+", text)
    if len(parts) > 1:
        result = []
        for part in parts[1:]:
            lines = part.split("\n", 1)
            result.append((lines[0].strip(), lines[1].strip() if len(lines) > 1 else ""))
        return result

    return [("📋 AI 건축 종합 크리틱", text)]


def _rich_text_chunks(text: str, limit: int = 1900) -> list[dict]:
    return [
        {"type": "text", "text": {"content": text[i : i + limit]}}
        for i in range(0, len(text), limit)
    ]


def _paragraph_blocks(text: str) -> list[dict]:
    chunks = [text[i : i + 1900] for i in range(0, len(text), 1900)]
    blocks = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": _rich_text_chunks(chunk)},
        }
        for chunk in chunks
    ]
    return blocks or [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": "(내용 없음)"}}]
            },
        }
    ]


def _toggle_heading_block(title: str, body: str) -> dict:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": title}}],
            "is_toggleable": True,
            "children": _paragraph_blocks(body),
        },
    }


def _build_notion_children(
    sections: list[tuple[str, str]], source_type: str, file_name: str
) -> list[dict]:
    blocks: list[dict] = [
        {
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"type": "text", "text": {"content": "📐 ArchivNotion 추론 리포트"}}]
            },
        }
    ]
    if source_type in ("사진/이미지 파일", "PDF 문서") and file_name:
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": f"📄 원본 파일: {file_name}  |  Gemini 네이티브 멀티모달 추론 완료"},
                }],
                "icon": {"type": "emoji", "emoji": "📁"},
            },
        })
    for title, body in sections:
        blocks.append(_toggle_heading_block(title, body))
    return blocks[:100]


# ════════════════════════════════════════════════════════════════════════════════
# 7. Streamlit UI — 페이지 설정 & 인증 게이트
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ArchivNotion", page_icon="📐", layout="centered")

if stauth is None:
    st.error("인증 시스템 구동 실패: `pip install streamlit-authenticator bcrypt` 를 실행하세요.")
    st.stop()

authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"],
)

# ── 인증 게이트 ───────────────────────────────────────────────────────────────

if not st.session_state.get("authentication_status"):

    st.title("📐 ArchivNotion")
    st.caption("건축 AI 에이전트 & 데이터 허브 아카이빙  |  Gemini Powered")
    st.divider()

    tab_login, tab_register = st.tabs(["🔐 로그인", "📝 회원가입"])

    with tab_login:
        try:
            login_result = authenticator.login(location="main")
        except TypeError:
            login_result = authenticator.login("🔐 ArchivNotion 로그인", "main")

        if isinstance(login_result, tuple):
            (
                st.session_state["name"],
                st.session_state["authentication_status"],
                st.session_state["username"],
            ) = login_result

        _status = st.session_state.get("authentication_status")
        if _status is False:
            st.error("사용자 ID 또는 비밀번호가 올바르지 않습니다.")
        elif _status is None:
            st.info("아이디와 비밀번호를 입력하여 로그인하세요.")

    with tab_register:
        try:
            try:
                reg = authenticator.register_user(location="main", pre_authorization=False)
            except TypeError:
                reg = authenticator.register_user("📝 회원가입", location="main", pre_authorization=False)

            if isinstance(reg, (tuple, list)) and reg and reg[0]:
                new_name = reg[2] if len(reg) > 2 else "신규 사용자"
                st.success(f"'{new_name}' 계정 등록 완료! 로그인 탭에서 로그인하세요.")
                _save_auth_config(auth_config)
            elif reg is True:
                st.success("회원가입 완료! 로그인 탭에서 로그인하세요.")
                _save_auth_config(auth_config)
        except Exception as _e:
            st.error(str(_e))

    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# 8. 메인 앱 — 인증 성공 시 진입
# ════════════════════════════════════════════════════════════════════════════════

if "project_map" not in st.session_state:
    st.session_state["project_map"] = dict(_DEFAULT_PROJECT_MAP)


# ── 사이드바 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**👤 {st.session_state.get('name', '사용자')}** 님 환영합니다")
    try:
        authenticator.logout(location="sidebar")
    except TypeError:
        authenticator.logout("🚪 로그아웃", "sidebar")

    st.divider()
    st.header("📁 프로젝트 폴더")

    project_keys = list(st.session_state["project_map"].keys())
    project_folder: str = st.selectbox(
        "자료를 귀속시킬 프로젝트를 선택하세요",
        project_keys,
    )

    st.markdown("**➕ 새 프로젝트 폴더 추가**")
    new_folder_name = st.text_input(
        "새 폴더명",
        placeholder="예: [졸업설계] 도시재생 복합주거",
        label_visibility="collapsed",
    )

    if st.button("📂 노션에 폴더 생성", use_container_width=True):
        name_clean = new_folder_name.strip()
        if not name_clean:
            st.warning("폴더명을 입력해주세요.")
        elif name_clean in st.session_state["project_map"]:
            st.warning("이미 존재하는 폴더명입니다.")
        else:
            with st.spinner("노션 프로젝트 페이지 생성 중..."):
                new_id, result_msg = create_notion_project(name_clean)
            if new_id:
                st.session_state["project_map"][name_clean] = new_id
                st.success(f"'{name_clean}' 폴더 생성 완료!")
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            else:
                st.error(result_msg)


# ── 메인 헤더 ─────────────────────────────────────────────────────────────────

st.title("📐 ArchivNotion")
st.subheader("건축 AI 에이전트 & 데이터 허브 아카이빙  |  Gemini Powered")
st.markdown("---")


# ── 자료 입력 ─────────────────────────────────────────────────────────────────

st.header("1. 리서치 소스 입력")
source_type: str = st.radio(
    "분석할 자료 형태", ["URL 링크", "사진/이미지 파일", "PDF 문서"]
)

source_url: str = ""
uploaded_file   = None

if source_type == "URL 링크":
    source_url = st.text_input("해외 매체 아티클 또는 논문 웹 URL 주소")
elif source_type == "사진/이미지 파일":
    uploaded_file = st.file_uploader(
        "건축 스케치, 사진 또는 도면 이미지 업로드", type=["jpg", "jpeg", "png"]
    )
elif source_type == "PDF 문서":
    uploaded_file = st.file_uploader(
        "해외 논문 원본 또는 PDF 아카이빙 파일 업로드", type=["pdf"]
    )


# ── 레이어 선택 ───────────────────────────────────────────────────────────────

st.header("2. 집중 크리틱 레이어 선택")
st.caption("AI가 집중적으로 가설을 수립하고 역추적할 관점을 선택하세요. (복수 선택 가능)")

col1, col2 = st.columns(2)
with col1:
    layer_structure = st.checkbox("🔍 [레이어 1] 구조적 발현 & 하중 흐름")
    layer_services  = st.checkbox("⚙️ [레이어 2] 설비 가시성 & 코어 시스템")
    layer_sustain   = st.checkbox("🌱 [레이어 3] 지속 가능성 & 에너지 루프")
with col2:
    layer_human    = st.checkbox("🚶 [레이어 4] 인간 스케일 & 가로 관계")
    layer_geometry = st.checkbox("📐 [레이어 5] 디자인 기하학 & 형태 언어")

# ── 커스텀 크리틱 레이어 ──────────────────────────────────────────────────────

st.markdown("---")
custom_layer: str = st.text_input(
    "✏️ 나만의 커스텀 크리틱 관점 추가",
    placeholder="예: 재료의 물성과 분절 디테일 / 빛과 그림자의 연출 전략 / 입면 패턴의 문화적 맥락",
    help="입력 시 Gemini가 해당 관점에 대한 별도 분석 섹션을 생성합니다.",
)


# ════════════════════════════════════════════════════════════════════════════════
# 9. 실행 버튼 — Gemini 추론 + Notion 전송 파이프라인
# ════════════════════════════════════════════════════════════════════════════════

if st.button("ArchivNotion 런칭 및 노션 전송", type="primary", use_container_width=True):

    if source_type == "URL 링크" and not source_url.strip():
        st.warning("유효한 URL 주소를 입력해 주세요.")
        st.stop()
    if source_type in ("사진/이미지 파일", "PDF 문서") and uploaded_file is None:
        st.warning("분석할 파일을 업로드해 주세요.")
        st.stop()

    file_bytes: bytes | None = uploaded_file.read() if uploaded_file else None
    file_name: str           = uploaded_file.name   if uploaded_file else ""

    layer_map: dict[str, bool] = {
        "🔍 [레이어 1] 구조적 발현 & 하중 흐름":  layer_structure,
        "⚙️ [레이어 2] 설비 가시성 & 코어 시스템": layer_services,
        "🌱 [레이어 3] 지속 가능성 & 에너지 루프": layer_sustain,
        "🚶 [레이어 4] 인간 스케일 & 가로 관계":  layer_human,
        "📐 [레이어 5] 디자인 기하학 & 형태 언어": layer_geometry,
    }
    selected_layers: list[str] = [name for name, checked in layer_map.items() if checked]

    has_custom = bool(custom_layer.strip())
    if has_custom:
        selected_layers.append(f"🎯 [사용자 커스텀 레이어]: {custom_layer.strip()}")

    layers_text: str = (
        "\n".join(f"- {l}" for l in selected_layers)
        if selected_layers
        else "- 기본 번역 요약 및 전체 레이어 균형 분석"
    )

    system_prompt = build_system_prompt(layers_text, has_custom)

    # ── Gemini 분석 ──────────────────────────────────────────────────────────
    with st.spinner("ArchivNotion 에이전트(Gemini)가 컨텍스트를 추론하고 있습니다..."):
        try:
            analysis_result: str = analyze_gemini(
                system_prompt, source_url, file_bytes, source_type, file_name
            )
        except Exception as e:
            st.error(f"Gemini 추론 엔진 가동 실패: {e}")
            st.stop()

    st.success("건축적 추론 완료!")
    st.markdown("### 📋 AI 건축 추론 리포트 프리뷰")
    st.markdown(analysis_result)

    # ── Notion 전송 ──────────────────────────────────────────────────────────
    with st.spinner("노션 프로젝트 폴더로 데이터 전송 중..."):
        page_title = (
            f"ArchivNotion — {source_url[:40]}" if source_url
            else f"ArchivNotion — {file_name}"
        )
        properties: dict = {
            "이름": {"title": [{"text": {"content": page_title}}]},
            "원본 소스": {"url": source_url if source_url else "https://archivnotion.ai"},
        }

        rel = _relation_prop(project_folder)
        if rel:
            properties[NOTION_RELATION_PROP] = rel

        sections = _parse_sections(analysis_result)
        notion_children = _build_notion_children(sections, source_type, file_name)

        payload: dict = {
            "parent": {"database_id": NOTION_MAIN_DB_ID},
            "properties": properties,
            "children": notion_children,
        }

        notion_res = requests.post(
            "https://api.notion.com/v1/pages",
            headers=_notion_headers(),
            json=payload,
        )

        if notion_res.status_code == 200:
            page_url = notion_res.json().get("url", "")
            st.balloons()
            link = f"[여기를 클릭해 노션 페이지 확인하기]({page_url})" if page_url else ""
            st.success(f"ArchivNotion 아카이빙 성공! {link}")
        else:
            st.error(f"노션 API 전송 실패 ({notion_res.status_code}): {notion_res.text}")
            