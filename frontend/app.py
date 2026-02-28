from __future__ import annotations
import os
import time
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="NotebookLM Clone", page_icon="📚", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 30

STATUS_LABELS = {
    "ready": "Ready",
    "failed": "Failed",
    "processing": "Processing",
    "pending": "Pending",
}

STATUS_ICONS = {
    "ready": "✓",
    "failed": "!",
    "processing": "~",
    "pending": "…",
}


def get_http_session() -> requests.Session:
    session = st.session_state.get("http_session")
    if isinstance(session, requests.Session):
        return session
    session = requests.Session()
    st.session_state["http_session"] = session
    return session


def api_request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json_payload: dict | None = None,
    data: dict[str, str | int] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
) -> tuple[bool, dict | list | str, int | None]:
    base_url = st.session_state.get("backend_url", BACKEND_URL).rstrip("/")
    url = f"{base_url}{path}"
    try:
        response = get_http_session().request(
            method=method,
            url=url,
            params=params,
            json=json_payload,
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        status_code = response.status_code
        if 200 <= status_code < 300:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return True, response.json(), status_code
            text_body = response.text.strip()
            return True, (text_body if text_body else "ok"), status_code

        try:
            error_body: dict | list | str = response.json()
        except ValueError:
            error_body = response.text
        return False, f"HTTP {status_code}: {error_body}", status_code
    except requests.RequestException as exc:
        return False, str(exc), None


def api_get(path: str, params: dict | None = None) -> tuple[bool, dict | list | str, int | None]:
    return api_request("GET", path, params=params)


def api_post(path: str, payload: dict) -> tuple[bool, dict | list | str, int | None]:
    return api_request("POST", path, json_payload=payload)


def api_patch(path: str, payload: dict) -> tuple[bool, dict | list | str, int | None]:
    return api_request("PATCH", path, json_payload=payload)


def api_delete(path: str) -> tuple[bool, dict | list | str, int | None]:
    return api_request("DELETE", path)


def api_get_bytes(path: str) -> tuple[bool, bytes | str, int | None]:
    base_url = st.session_state.get("backend_url", BACKEND_URL).rstrip("/")
    url = f"{base_url}{path}"
    try:
        response = get_http_session().request(
            method="GET",
            url=url,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return False, str(exc), None

    if 200 <= response.status_code < 300:
        return True, response.content, response.status_code
    return False, f"HTTP {response.status_code}: {response.text}", response.status_code


def api_post_multipart(
    path: str,
    data: dict[str, str | int],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bool, dict | list | str, int | None]:
    return api_request("POST", path, data=data, files=files)


def fetch_notebooks() -> tuple[bool, list[dict] | str]:
    ok, result, _ = api_get("/notebooks")
    if not ok:
        return False, str(result)
    notebooks = result if isinstance(result, list) else []
    return True, notebooks


def get_query_param(name: str) -> str | None:
    try:
        value = st.query_params.get(name)
    except Exception:
        params = st.experimental_get_query_params()
        raw = params.get(name)
        if isinstance(raw, list):
            return str(raw[0]) if raw else None
        if raw is None:
            return None
        return str(raw)

    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None:
        return None
    return str(value)


def remove_query_param(name: str) -> None:
    try:
        if name in st.query_params:
            del st.query_params[name]
        return
    except Exception:
        pass

    params = st.experimental_get_query_params()
    if name in params:
        params.pop(name, None)
        st.experimental_set_query_params(**params)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

            :root {
                --ink: #162525;
                --ink-muted: #4f6362;
                --card: #fefdf9;
                --card-border: #d2dad3;
                --accent: #146f67;
                --accent-soft: #d6efeb;
                --warn-soft: #fff2dd;
                --warn-border: #f0bd69;
                --ok-soft: #def5ec;
                --ok-border: #53a27f;
                --error-soft: #ffe6e3;
                --error-border: #d2675a;
                --pending-soft: #eceeef;
                --pending-border: #9aa7ad;
            }

            .stApp {
                background:
                    radial-gradient(circle at 15% -5%, #fff3de 0%, rgba(255, 243, 222, 0) 40%),
                    radial-gradient(circle at 88% 8%, #d9efeb 0%, rgba(217, 239, 235, 0) 42%),
                    linear-gradient(180deg, #f8faf8 0%, #f1f5f2 100%);
                color: var(--ink);
                font-family: "IBM Plex Sans", sans-serif;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #173436 0%, #11292b 100%);
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] small,
            [data-testid="stSidebar"] span {
                color: #e9f1ef;
                font-family: "IBM Plex Sans", sans-serif;
            }

            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stTextArea textarea,
            [data-testid="stSidebar"] div[data-baseweb="select"] > div,
            [data-testid="stSidebar"] .stNumberInput input {
                background: rgba(233, 241, 239, 0.1);
                color: #f3f8f7;
                border: 1px solid rgba(178, 209, 202, 0.35);
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] * {
                color: #0f2a27;
            }

            h1, h2, h3, h4 {
                font-family: "Space Grotesk", sans-serif;
                letter-spacing: 0.02em;
                color: var(--ink);
            }

            .app-hero {
                background: linear-gradient(125deg, rgba(20, 111, 103, 0.13) 0%, rgba(255, 245, 228, 0.95) 56%, rgba(205, 231, 226, 0.9) 100%);
                border: 1px solid #c9d9cf;
                border-radius: 18px;
                padding: 18px 22px;
                margin-bottom: 14px;
                box-shadow: 0 8px 24px rgba(17, 45, 42, 0.08);
            }

            .hero-kicker {
                display: inline-block;
                font-family: "Space Grotesk", sans-serif;
                font-size: 0.75rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #0f625b;
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(15, 98, 91, 0.22);
                border-radius: 999px;
                padding: 3px 10px;
                margin-bottom: 8px;
            }

            .hero-title {
                margin: 0;
                font-family: "Space Grotesk", sans-serif;
                font-size: clamp(1.45rem, 2.4vw, 2rem);
                line-height: 1.2;
            }

            .hero-sub {
                margin: 6px 0 0 0;
                color: var(--ink-muted);
                font-size: 0.95rem;
            }

            .soft-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 12px 14px;
            }

            .metric-card {
                background: var(--card);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 10px 12px;
            }

            .metric-label {
                color: var(--ink-muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .metric-value {
                font-family: "Space Grotesk", sans-serif;
                font-size: 1.4rem;
                color: var(--ink);
                margin-top: 2px;
            }

            .status-pill {
                display: inline-block;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.78rem;
                font-weight: 600;
                border: 1px solid transparent;
            }
            .status-ready {
                background: var(--ok-soft);
                border-color: var(--ok-border);
                color: #1c6b4d;
            }
            .status-failed {
                background: var(--error-soft);
                border-color: var(--error-border);
                color: #8d3329;
            }
            .status-processing {
                background: var(--warn-soft);
                border-color: var(--warn-border);
                color: #7d581a;
            }
            .status-pending {
                background: var(--pending-soft);
                border-color: var(--pending-border);
                color: #485a63;
            }

            div.stButton > button,
            div.stFormSubmitButton > button,
            div.stDownloadButton > button {
                border-radius: 11px;
                border: 1px solid #0f5f57;
                background: linear-gradient(135deg, #167269 0%, #0f5f57 100%);
                color: #f4fffd !important;
                font-weight: 600;
                padding: 0.38rem 0.95rem;
                transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
                box-shadow: 0 3px 10px rgba(15, 95, 87, 0.22);
            }

            div.stButton > button:hover,
            div.stFormSubmitButton > button:hover,
            div.stDownloadButton > button:hover {
                transform: translateY(-1px);
                filter: brightness(1.03);
                box-shadow: 0 5px 14px rgba(15, 95, 87, 0.28);
            }

            div.stButton > button:focus,
            div.stFormSubmitButton > button:focus,
            div.stDownloadButton > button:focus {
                outline: 2px solid #ffcb7f;
                outline-offset: 2px;
            }

            [data-testid="stSidebar"] div.stButton > button,
            [data-testid="stSidebar"] div.stFormSubmitButton > button,
            [data-testid="stSidebar"] div.stDownloadButton > button {
                background: linear-gradient(135deg, #ecf8f4 0%, #d8efe9 100%);
                color: #123f3b !important;
                border: 1px solid #90c6bb;
                box-shadow: none;
            }

            [data-testid="stSidebar"] div.stButton > button:hover,
            [data-testid="stSidebar"] div.stFormSubmitButton > button:hover,
            [data-testid="stSidebar"] div.stDownloadButton > button:hover {
                background: linear-gradient(135deg, #f5fffc 0%, #e3f8f3 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_key(value: str | None) -> str:
    return str(value or "").strip().lower()


def render_status_pill(status: str | None, *, prefix: str = "") -> None:
    key = status_key(status)
    css = f"status-{key}" if key in STATUS_LABELS else "status-pending"
    icon = STATUS_ICONS.get(key, "•")
    label = STATUS_LABELS.get(key, str(status or "Unknown").title())
    text = f"{prefix}{icon} {label}".strip()
    st.markdown(f"<span class='status-pill {css}'>{text}</span>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str | int) -> None:
    st.markdown(
        (
            "<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def build_artifact_option_label(artifact: dict[str, Any]) -> str:
    art_id = artifact.get("id", "?")
    art_type = str(artifact.get("type", "artifact")).title()
    status = status_key(str(artifact.get("status", "")))
    icon = STATUS_ICONS.get(status, "•")
    label = STATUS_LABELS.get(status, str(artifact.get("status", "unknown")).title())
    return f"#{art_id} · {art_type} · {icon} {label}"


inject_theme()

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "attempted_auto_dev_login" not in st.session_state:
    st.session_state["attempted_auto_dev_login"] = False
if "bridge_error" not in st.session_state:
    st.session_state["bridge_error"] = None
if "processed_bridge_token" not in st.session_state:
    st.session_state["processed_bridge_token"] = None

bridge_token = get_query_param("auth_bridge")
if bridge_token and bridge_token != st.session_state["processed_bridge_token"]:
    st.session_state["processed_bridge_token"] = bridge_token
    bridge_ok, bridge_result, _ = api_post("/auth/bridge/exchange", {"token": bridge_token})
    remove_query_param("auth_bridge")
    if bridge_ok:
        st.session_state["bridge_error"] = None
        st.rerun()
    st.session_state["bridge_error"] = str(bridge_result)
    st.rerun()

with st.sidebar:
    st.markdown("### Workspace")
    st.caption("Connected client settings")
    st.text_input("Backend URL", value=BACKEND_URL, key="backend_url")

    auth_ok, auth_result, _ = api_get("/auth/status")
    auth_data = auth_result if (auth_ok and isinstance(auth_result, dict)) else {}
    auth_mode = str(auth_data.get("mode", "unknown"))
    authenticated = bool(auth_data.get("authenticated"))
    auth_user = auth_data.get("user") if isinstance(auth_data.get("user"), dict) else None

    if auth_mode == "dev" and not authenticated and not st.session_state["attempted_auto_dev_login"]:
        st.session_state["attempted_auto_dev_login"] = True
        login_ok, _, _ = api_post("/auth/dev-login", {})
        if login_ok:
            st.rerun()

    st.subheader("Authentication")
    st.caption(f"Mode: {auth_mode}")
    bridge_error = st.session_state.get("bridge_error")
    if bridge_error:
        st.error(f"OAuth bridge failed: {bridge_error}")
        st.session_state["bridge_error"] = None

    if not auth_ok:
        st.error(f"Failed to reach backend auth endpoint: {auth_result}")
    elif authenticated and auth_user:
        st.success(f"Signed in as {auth_user.get('email', 'unknown')}.")
        if st.button("Sign out"):
            api_post("/auth/logout", {})
            st.session_state.pop("selected_notebook_id", None)
            st.session_state.pop("selected_notebook_title", None)
            st.session_state.pop("selected_thread_id", None)
            st.rerun()
    elif auth_mode == "dev":
        with st.form("dev_login_form"):
            email = st.text_input("Email", value="dev@example.com")
            display_name = st.text_input("Display name", value="Dev User")
            login_submitted = st.form_submit_button("Sign in")
        if login_submitted:
            login_ok, login_result, _ = api_post(
                "/auth/dev-login",
                {
                    "email": email.strip() or None,
                    "display_name": display_name.strip() or None,
                },
            )
            if login_ok:
                st.rerun()
            st.error(f"Login failed: {login_result}")
    elif auth_mode == "hf_oauth":
        backend_root = st.session_state.get("backend_url", BACKEND_URL).rstrip("/")
        st.markdown(f"[Sign in with Hugging Face]({backend_root}/auth/login)")
        st.caption("After provider login, you will be redirected back and signed in automatically.")
        if st.button("Refresh auth"):
            st.rerun()
    else:
        st.warning("Authentication is not configured.")

    st.divider()
    page_options = ["Home", "Notebooks"]
    default_page = st.session_state.get("page", "Home")
    default_index = page_options.index(default_page) if default_page in page_options else 0
    page = st.radio("Go to", page_options, index=default_index)
    st.session_state["page"] = page

    selected_notebook_id = st.session_state.get("selected_notebook_id")
    selected_notebook_title = st.session_state.get("selected_notebook_title")
    if selected_notebook_id:
        st.caption(f"Current notebook: {selected_notebook_title} (ID: {selected_notebook_id})")
        if st.button("Clear notebook selection"):
            st.session_state.pop("selected_notebook_id", None)
            st.session_state.pop("selected_notebook_title", None)
            st.session_state.pop("selected_thread_id", None)
            st.rerun()

hero_mode = auth_mode if isinstance(auth_mode, str) else "unknown"
hero_identity = "Not signed in"
if authenticated and isinstance(auth_user, dict):
    hero_identity = str(auth_user.get("display_name") or auth_user.get("email") or "Signed in")
st.markdown(
    (
        "<div class='app-hero'>"
        "<span class='hero-kicker'>Notebook Workspace</span>"
        "<h1 class='hero-title'>NotebookLM Clone</h1>"
        "<p class='hero-sub'>"
        "Ingest sources, chat with citations, and generate reports, quizzes, and podcasts."
        f" &nbsp;|&nbsp; Auth mode: <strong>{hero_mode}</strong>"
        f" &nbsp;|&nbsp; User: <strong>{hero_identity}</strong>"
        "</p>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

if not auth_ok:
    st.error(f"Backend auth check failed: {auth_result}")
    st.stop()

if not authenticated:
    st.info("Sign in to continue.")
    st.stop()

if page == "Home":
    st.subheader("Home")
    st.markdown(
        "<div class='soft-card'>Use this page to verify backend connectivity and auth status before working in notebooks.</div>",
        unsafe_allow_html=True,
    )

    ok, result, _ = api_get("/health")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        render_metric_card("Auth Mode", hero_mode)
    with metric_col_2:
        render_metric_card("Session", "Signed in" if authenticated else "Signed out")
    with metric_col_3:
        render_metric_card("Backend", "Healthy" if ok else "Unavailable")

    if ok:
        st.success("Backend health check passed.")
        st.json(result)
    else:
        st.error("Backend health check failed.")
        st.code(str(result))

elif page == "Notebooks":
    st.subheader("Notebooks")
    st.markdown(
        "<div class='soft-card'>Create or open a notebook, ingest sources, then use chat and artifact tools from one workspace.</div>",
        unsafe_allow_html=True,
    )

    create_col, refresh_col = st.columns([3, 1])
    with create_col:
        with st.form("create_notebook_form"):
            notebook_title = st.text_input("Notebook title", placeholder="e.g., AI Research Notes")
            submitted = st.form_submit_button("Create notebook")
    with refresh_col:
        st.write("")
        st.write("")
        if st.button("Refresh notebooks", use_container_width=True):
            st.rerun()

    if submitted:
        if notebook_title.strip():
            payload = {
                "title": notebook_title.strip(),
            }
            ok, result, _ = api_post("/notebooks", payload)
            if ok:
                st.success("Notebook created.")
                st.json(result)
            else:
                st.error("Failed to create notebook.")
                st.code(str(result))
        else:
            st.error("Notebook title is required.")

    ok, result = fetch_notebooks()
    if ok:
        notebooks = result if isinstance(result, list) else []
        selected_notebook_id = st.session_state.get("selected_notebook_id")
        selected_notebook_title = st.session_state.get("selected_notebook_title")

        metrics_left, metrics_mid, metrics_right = st.columns(3)
        with metrics_left:
            render_metric_card("Notebook Count", len(notebooks))
        with metrics_mid:
            render_metric_card("Selected Notebook", selected_notebook_id or "None")
        with metrics_right:
            render_metric_card("Workspace State", "Ready" if notebooks else "Empty")

        if notebooks:
            st.write("Your notebooks")
            st.dataframe(notebooks, use_container_width=True, hide_index=True)

            notebook_options = {
                f"{n['id']} - {n['title']}": n
                for n in notebooks
                if isinstance(n, dict) and "id" in n and "title" in n
            }
            labels = list(notebook_options.keys())
            default_index = 0
            for idx, label in enumerate(labels):
                if notebook_options[label]["id"] == selected_notebook_id:
                    default_index = idx
                    break
            selected_label = st.selectbox("Select notebook to open", options=labels, index=default_index)
            selected = notebook_options[selected_label]
            if st.button("Open notebook"):
                st.session_state["selected_notebook_id"] = selected["id"]
                st.session_state["selected_notebook_title"] = selected["title"]
                st.session_state["page"] = "Notebooks"
                st.rerun()

            selected_notebook_id = st.session_state.get("selected_notebook_id")
            selected_notebook_title = st.session_state.get("selected_notebook_title")
            if selected_notebook_id:
                st.divider()
                st.subheader(f"Notebook: {selected_notebook_title}")
                render_status_pill("ready", prefix="Notebook ")

                manage_left, manage_right = st.columns(2)
                with manage_left:
                    with st.form("rename_notebook_form"):
                        renamed_title = st.text_input(
                            "Rename notebook",
                            value=selected_notebook_title or "",
                            key=f"rename_notebook_title_{selected_notebook_id}",
                        )
                        rename_submitted = st.form_submit_button("Save name")
                    if rename_submitted:
                        if not renamed_title.strip():
                            st.error("Notebook title cannot be empty.")
                        else:
                            ok, rename_result, _ = api_patch(
                                f"/notebooks/{selected_notebook_id}",
                                {"title": renamed_title.strip()},
                            )
                            if ok and isinstance(rename_result, dict):
                                st.success("Notebook renamed.")
                                st.session_state["selected_notebook_title"] = rename_result.get(
                                    "title", renamed_title.strip()
                                )
                                st.rerun()
                            st.error("Failed to rename notebook.")
                            st.code(str(rename_result))

                with manage_right:
                    with st.form("delete_notebook_form"):
                        confirm_delete = st.checkbox(
                            "I understand this permanently deletes this notebook and its data.",
                            value=False,
                        )
                        delete_submitted = st.form_submit_button("Delete notebook")
                    if delete_submitted:
                        if not confirm_delete:
                            st.error("Please confirm deletion first.")
                        else:
                            ok, delete_result, _ = api_delete(f"/notebooks/{selected_notebook_id}")
                            if ok:
                                st.success("Notebook deleted.")
                                st.session_state.pop("selected_notebook_id", None)
                                st.session_state.pop("selected_notebook_title", None)
                                st.session_state.pop("selected_thread_id", None)
                                st.rerun()
                            st.error("Failed to delete notebook.")
                            st.code(str(delete_result))

                source_tab, chat_tab, artifacts_tab = st.tabs(
                    ["Source Library", "Chat Workspace", "Artifact Studio"]
                )

                with source_tab:
                    st.markdown(
                        "<div class='soft-card'>Upload files or add links. Ingested sources are scoped to this notebook.</div>",
                        unsafe_allow_html=True,
                    )
                    ok, notebook_result = fetch_notebooks()
                    if not ok:
                        st.error("Failed to load notebooks.")
                        st.code(str(notebook_result))
                    else:
                        notebooks_for_user = notebook_result if isinstance(notebook_result, list) else []
                        notebook_ids = {
                            n["id"]
                            for n in notebooks_for_user
                            if isinstance(n, dict) and "id" in n
                        }
                        if selected_notebook_id not in notebook_ids:
                            st.error("Selected notebook is not available for this user.")
                        else:
                            with st.form("create_source_form"):
                                file_like_types = ["file", "pdf", "pptx", "txt", "md", "docx"]
                                source_type = st.selectbox(
                                    "Source type", options=file_like_types + ["url", "text"], index=0
                                )
                                source_title = st.text_input("Source title (optional)")
                                original_name = st.text_input("Original filename (optional)")
                                source_url = st.text_input("Source URL (optional)")
                                storage_path = st.text_input("Storage path (optional)")
                                uploaded_file = st.file_uploader(
                                    "Upload file (used for type=file)",
                                    type=["pdf", "pptx", "txt", "md", "docx"],
                                )
                                source_status = st.selectbox(
                                    "Status",
                                    options=["pending", "processing", "ready", "failed"],
                                    index=0,
                                )
                                source_submitted = st.form_submit_button("Add source")

                            if source_submitted:
                                resolved_original_name = original_name or None
                                resolved_title = source_title or None

                                if source_type in file_like_types:
                                    if uploaded_file is None:
                                        st.error("Please upload a file for this source type.")
                                        st.stop()
                                    resolved_original_name = resolved_original_name or uploaded_file.name
                                    resolved_title = resolved_title or uploaded_file.name

                                    form_data: dict[str, str | int] = {"status": source_status}
                                    if resolved_title:
                                        form_data["title"] = resolved_title
                                    files_payload = {
                                        "file": (
                                            uploaded_file.name,
                                            uploaded_file.getvalue(),
                                            uploaded_file.type or "application/octet-stream",
                                        )
                                    }
                                    ok, create_result, _ = api_post_multipart(
                                        f"/notebooks/{selected_notebook_id}/sources/upload",
                                        data=form_data,
                                        files=files_payload,
                                    )
                                    if ok:
                                        st.success("File uploaded and source added.")
                                        st.json(create_result)
                                    else:
                                        st.error("Failed to upload file source.")
                                        st.code(str(create_result))
                                    st.rerun()

                                if source_type == "url" and not source_url.strip():
                                    st.error("Please provide a URL when source type is 'url'.")
                                    st.stop()

                                if source_type not in file_like_types:
                                    payload = {
                                        "type": source_type,
                                        "title": resolved_title,
                                        "original_name": resolved_original_name,
                                        "url": source_url or None,
                                        "storage_path": storage_path or None,
                                        "status": source_status,
                                    }
                                    ok, create_result, _ = api_post(
                                        f"/notebooks/{selected_notebook_id}/sources", payload
                                    )
                                    if ok:
                                        st.success("Source added.")
                                        st.json(create_result)
                                    else:
                                        st.error("Failed to add source.")
                                        st.code(str(create_result))

                            if st.button("Refresh sources"):
                                st.rerun()

                            ok, source_result, _ = api_get(
                                f"/notebooks/{selected_notebook_id}/sources",
                            )
                            if ok:
                                sources = source_result if isinstance(source_result, list) else []
                                if sources:
                                    ready_sources = sum(
                                        1 for src in sources if status_key(str(src.get("status", ""))) == "ready"
                                    )
                                    processing_sources = sum(
                                        1
                                        for src in sources
                                        if status_key(str(src.get("status", ""))) in {"processing", "pending"}
                                    )
                                    source_metrics_1, source_metrics_2, source_metrics_3 = st.columns(3)
                                    with source_metrics_1:
                                        render_metric_card("Source Count", len(sources))
                                    with source_metrics_2:
                                        render_metric_card("Ready Sources", ready_sources)
                                    with source_metrics_3:
                                        render_metric_card("In Flight", processing_sources)
                                    st.write("Sources")
                                    st.dataframe(sources, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No sources yet for this notebook.")
                            else:
                                st.error("Failed to fetch sources.")
                                st.code(str(source_result))

                with chat_tab:
                    st.markdown(
                        "<div class='soft-card'>Ask questions grounded in your selected notebook sources. Citations are shown per assistant response.</div>",
                        unsafe_allow_html=True,
                    )
                    st.write("Chat threads")
                    ok, thread_result, _ = api_get(
                        f"/notebooks/{selected_notebook_id}/threads",
                    )
                    threads = thread_result if (ok and isinstance(thread_result, list)) else []
                    render_metric_card("Thread Count", len(threads))

                    with st.form("create_thread_form"):
                        thread_title = st.text_input("Thread title (optional)")
                        create_thread_submitted = st.form_submit_button("Create thread")

                    if create_thread_submitted:
                        payload = {
                            "title": thread_title.strip() or None,
                        }
                        ok, create_thread_result, _ = api_post(
                            f"/notebooks/{selected_notebook_id}/threads", payload
                        )
                        if ok and isinstance(create_thread_result, dict):
                            st.success("Thread created.")
                            st.session_state["selected_thread_id"] = create_thread_result["id"]
                            st.rerun()
                        else:
                            st.error("Failed to create thread.")
                            st.code(str(create_thread_result))

                    if threads:
                        thread_options = {
                            f"{t['id']} - {t.get('title') or 'Untitled'}": t["id"] for t in threads
                        }
                        thread_labels = list(thread_options.keys())
                        selected_thread_id = st.session_state.get("selected_thread_id")
                        default_thread_index = 0
                        if selected_thread_id:
                            for idx, label in enumerate(thread_labels):
                                if thread_options[label] == selected_thread_id:
                                    default_thread_index = idx
                                    break
                        selected_thread_label = st.selectbox(
                            "Select thread",
                            options=thread_labels,
                            index=default_thread_index,
                            key="selected_thread_label",
                        )
                        selected_thread_id = thread_options[selected_thread_label]
                        st.session_state["selected_thread_id"] = selected_thread_id

                        ok, message_result, _ = api_get(
                            f"/threads/{selected_thread_id}/messages",
                            params={"notebook_id": selected_notebook_id},
                        )
                        if ok and isinstance(message_result, list):
                            st.write("Messages")
                            for msg in message_result:
                                role = msg.get("role", "unknown")
                                content = msg.get("content", "")
                                citations = msg.get("citations", [])
                                message_type = "assistant" if role == "assistant" else "user"
                                with st.chat_message(message_type):
                                    st.markdown(str(content))
                                    if role == "assistant" and isinstance(citations, list) and citations:
                                        with st.expander("Citations", expanded=False):
                                            st.dataframe(citations, use_container_width=True, hide_index=True)
                        else:
                            st.error("Failed to fetch messages.")
                            st.code(str(message_result))

                        with st.form("chat_form", clear_on_submit=True):
                            question = st.text_input("Ask a question")
                            ask_submitted = st.form_submit_button("Send")

                        if ask_submitted:
                            if not question.strip():
                                st.error("Question cannot be empty.")
                            else:
                                payload = {
                                    "question": question.strip(),
                                    "top_k": 5,
                                }
                                ok, chat_result, _ = api_post(
                                    f"/threads/{selected_thread_id}/chat?notebook_id={selected_notebook_id}",
                                    payload,
                                )
                                if ok and isinstance(chat_result, dict):
                                    st.success("Response generated.")
                                    citations = chat_result.get("citations", [])
                                    if citations:
                                        st.write("Citations")
                                        st.dataframe(citations, use_container_width=True)
                                    st.rerun()
                                else:
                                    st.error("Chat request failed.")
                                    st.code(str(chat_result))
                    else:
                        st.info("Create a thread to start chatting.")

                with artifacts_tab:
                    st.markdown(
                        "<div class='soft-card'>Generate structured outputs from notebook context and track progress here.</div>",
                        unsafe_allow_html=True,
                    )

                    report_col, quiz_col, podcast_col = st.columns(3)

                    with report_col:
                        with st.form("generate_report_form"):
                            report_title = st.text_input(
                                "Report title (optional)",
                                key=f"report_title_{selected_notebook_id}",
                            )
                            report_detail = st.selectbox(
                                "Detail level",
                                options=["short", "medium", "long"],
                                index=1,
                                key=f"report_detail_{selected_notebook_id}",
                            )
                            report_topic = st.text_input(
                                "Topic focus (optional)",
                                key=f"report_topic_{selected_notebook_id}",
                            )
                            report_submitted = st.form_submit_button("Generate report")

                        if report_submitted:
                            payload = {
                                "title": report_title.strip() or None,
                                "detail_level": report_detail,
                                "topic_focus": report_topic.strip() or None,
                            }
                            ok, report_result, _ = api_post(
                                f"/notebooks/{selected_notebook_id}/artifacts/report",
                                payload,
                            )
                            if ok:
                                st.success("Report generated.")
                                st.rerun()
                            else:
                                st.error("Report generation failed.")
                                st.code(str(report_result))

                    with quiz_col:
                        with st.form("generate_quiz_form"):
                            quiz_title = st.text_input(
                                "Quiz title (optional)",
                                key=f"quiz_title_{selected_notebook_id}",
                            )
                            quiz_questions = st.number_input(
                                "Questions",
                                min_value=1,
                                max_value=20,
                                value=5,
                                step=1,
                                key=f"quiz_questions_{selected_notebook_id}",
                            )
                            quiz_difficulty = st.selectbox(
                                "Difficulty",
                                options=["easy", "medium", "hard"],
                                index=1,
                                key=f"quiz_difficulty_{selected_notebook_id}",
                            )
                            quiz_topic = st.text_input(
                                "Topic focus (optional)",
                                key=f"quiz_topic_{selected_notebook_id}",
                            )
                            quiz_submitted = st.form_submit_button("Generate quiz")

                        if quiz_submitted:
                            payload = {
                                "title": quiz_title.strip() or None,
                                "num_questions": int(quiz_questions),
                                "difficulty": quiz_difficulty,
                                "topic_focus": quiz_topic.strip() or None,
                            }
                            ok, quiz_result, _ = api_post(
                                f"/notebooks/{selected_notebook_id}/artifacts/quiz",
                                payload,
                            )
                            if ok:
                                st.success("Quiz generated.")
                                st.rerun()
                            else:
                                st.error("Quiz generation failed.")
                                st.code(str(quiz_result))

                    with podcast_col:
                        with st.form("generate_podcast_form"):
                            podcast_title = st.text_input(
                                "Podcast title (optional)",
                                key=f"podcast_title_{selected_notebook_id}",
                            )
                            podcast_duration = st.selectbox(
                                "Duration",
                                options=["5min", "10min", "15min", "20min"],
                                index=0,
                                key=f"podcast_duration_{selected_notebook_id}",
                            )
                            podcast_topic = st.text_input(
                                "Topic focus (optional)",
                                key=f"podcast_topic_{selected_notebook_id}",
                            )
                            podcast_submitted = st.form_submit_button("Generate podcast")

                        if podcast_submitted:
                            payload = {
                                "title": podcast_title.strip() or None,
                                "duration": podcast_duration,
                                "topic_focus": podcast_topic.strip() or None,
                            }
                            ok, podcast_result, _ = api_post(
                                f"/notebooks/{selected_notebook_id}/artifacts/podcast",
                                payload,
                            )
                            if ok:
                                st.success("Podcast generation started.")
                                st.rerun()
                            else:
                                st.error("Podcast generation failed.")
                                st.code(str(podcast_result))

                    st.divider()
                    if st.button("Refresh artifacts"):
                        st.rerun()

                    ok, artifact_result, _ = api_get(f"/notebooks/{selected_notebook_id}/artifacts")
                    if ok and isinstance(artifact_result, list):
                        artifacts = artifact_result
                        if artifacts:
                            auto_refresh_key = f"auto_refresh_artifacts_{selected_notebook_id}"
                            auto_refresh = st.checkbox(
                                "Auto-refresh while artifacts are processing",
                                value=bool(st.session_state.get(auto_refresh_key, True)),
                                key=auto_refresh_key,
                            )
                            ready_count = sum(
                                1 for artifact in artifacts if status_key(str(artifact.get("status", ""))) == "ready"
                            )
                            failed_count = sum(
                                1 for artifact in artifacts if status_key(str(artifact.get("status", ""))) == "failed"
                            )
                            in_flight = sum(
                                1
                                for artifact in artifacts
                                if status_key(str(artifact.get("status", ""))) in {"pending", "processing"}
                            )
                            artifact_metric_1, artifact_metric_2, artifact_metric_3, artifact_metric_4 = st.columns(4)
                            with artifact_metric_1:
                                render_metric_card("Artifacts", len(artifacts))
                            with artifact_metric_2:
                                render_metric_card("Ready", ready_count)
                            with artifact_metric_3:
                                render_metric_card("Processing", in_flight)
                            with artifact_metric_4:
                                render_metric_card("Failed", failed_count)

                            st.dataframe(artifacts, use_container_width=True, hide_index=True)
                            artifact_options = {
                                build_artifact_option_label(a): a
                                for a in artifacts
                                if isinstance(a, dict) and "id" in a
                            }
                            selected_artifact_label = st.selectbox(
                                "Select artifact",
                                options=list(artifact_options.keys()),
                                key="selected_artifact_label",
                            )
                            selected_artifact = artifact_options[selected_artifact_label]
                            artifact_id = int(selected_artifact["id"])
                            artifact_type = str(selected_artifact.get("type", ""))
                            artifact_status = str(selected_artifact.get("status", ""))
                            artifact_content = selected_artifact.get("content")
                            artifact_error = selected_artifact.get("error_message")

                            status_col, type_col = st.columns([1, 4])
                            with status_col:
                                render_status_pill(artifact_status)
                            with type_col:
                                st.caption(f"Artifact type: {artifact_type}")

                            if artifact_error:
                                st.error(str(artifact_error))

                            if artifact_type == "report" and artifact_content:
                                st.markdown("### Report Preview")
                                st.markdown(str(artifact_content))
                                st.download_button(
                                    "Download report (.md)",
                                    data=str(artifact_content).encode("utf-8"),
                                    file_name=f"report_{artifact_id}.md",
                                    mime="text/markdown",
                                )
                            elif artifact_type == "quiz" and artifact_content:
                                st.markdown("### Quiz Preview")
                                st.markdown(str(artifact_content))
                                st.download_button(
                                    "Download quiz (.md)",
                                    data=str(artifact_content).encode("utf-8"),
                                    file_name=f"quiz_{artifact_id}.md",
                                    mime="text/markdown",
                                )
                            elif artifact_type == "podcast":
                                if artifact_content:
                                    st.markdown("### Transcript")
                                    st.markdown(str(artifact_content))
                                    st.download_button(
                                        "Download transcript (.md)",
                                        data=str(artifact_content).encode("utf-8"),
                                        file_name=f"podcast_transcript_{artifact_id}.md",
                                        mime="text/markdown",
                                    )
                                if artifact_status == "ready":
                                    ok_audio, audio_result, _ = api_get_bytes(
                                        f"/notebooks/{selected_notebook_id}/artifacts/{artifact_id}/audio"
                                    )
                                    if ok_audio and isinstance(audio_result, bytes):
                                        st.audio(audio_result, format="audio/mp3")
                                        st.download_button(
                                            "Download podcast (.mp3)",
                                            data=audio_result,
                                            file_name=f"podcast_{artifact_id}.mp3",
                                            mime="audio/mpeg",
                                        )
                                    else:
                                        st.error(f"Unable to load audio: {audio_result}")
                                else:
                                    st.info(f"Podcast status: {artifact_status}")
                            else:
                                st.info("Select an artifact to preview.")

                            if auto_refresh and in_flight > 0:
                                st.caption(
                                    f"{in_flight} artifact(s) still processing. "
                                    "Refreshing in 4 seconds..."
                                )
                                time.sleep(4)
                                st.rerun()
                        else:
                            st.info("No artifacts generated yet.")
                    else:
                        st.error("Failed to fetch artifacts.")
                        st.code(str(artifact_result))
        else:
            st.info("No notebooks yet.")
    else:
        st.error("Failed to fetch notebooks.")
        st.code(str(result))
