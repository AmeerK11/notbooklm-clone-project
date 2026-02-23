from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="NotebookLM Clone", page_icon="📚", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 30


def api_get(path: str, params: dict | None = None) -> tuple[bool, dict | list | str]:
    base_url = st.session_state.get("backend_url", BACKEND_URL).rstrip("/")
    try:
        response = requests.get(
            f"{base_url}{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def api_post(path: str, payload: dict) -> tuple[bool, dict | list | str]:
    base_url = st.session_state.get("backend_url", BACKEND_URL).rstrip("/")
    try:
        response = requests.post(
            f"{base_url}{path}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def api_post_multipart(
    path: str, data: dict[str, str | int], files: dict[str, tuple[str, bytes, str]]
) -> tuple[bool, dict | list | str]:
    base_url = st.session_state.get("backend_url", BACKEND_URL).rstrip("/")
    try:
        response = requests.post(
            f"{base_url}{path}",
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def fetch_notebooks(owner_user_id: int) -> tuple[bool, list[dict] | str]:
    ok, result = api_get("/notebooks", params={"owner_user_id": owner_user_id})
    if not ok:
        return False, str(result)
    notebooks = result if isinstance(result, list) else []
    return True, notebooks


st.title("NotebookLM Clone")
st.caption("Streamlit frontend shell")

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

with st.sidebar:
    st.header("Navigation")
    page_options = ["Home", "Notebooks"]
    default_page = st.session_state.get("page", "Home")
    default_index = page_options.index(default_page) if default_page in page_options else 0
    page = st.radio("Go to", page_options, index=default_index)
    st.session_state["page"] = page
    st.divider()
    st.text_input("Backend URL", value=BACKEND_URL, key="backend_url")
    st.number_input("Owner User ID", min_value=1, step=1, value=1, key="owner_user_id")
    selected_notebook_id = st.session_state.get("selected_notebook_id")
    selected_notebook_title = st.session_state.get("selected_notebook_title")
    if selected_notebook_id:
        st.caption(f"Current notebook: {selected_notebook_title} (ID: {selected_notebook_id})")
        if st.button("Clear notebook selection"):
            st.session_state.pop("selected_notebook_id", None)
            st.session_state.pop("selected_notebook_title", None)
            st.rerun()


if page == "Home":
    st.subheader("Home")
    st.write("Frontend is connected to your FastAPI backend.")

    ok, result = api_get("/health")
    if ok:
        st.success("Backend health check passed.")
        st.json(result)
    else:
        st.error("Backend health check failed.")
        st.code(str(result))

elif page == "Notebooks":
    st.subheader("Notebooks")
    owner_user_id = int(st.session_state["owner_user_id"])

    with st.form("create_notebook_form"):
        notebook_title = st.text_input("Notebook title", placeholder="e.g., AI Research Notes")
        submitted = st.form_submit_button("Create notebook")

    if submitted:
        if notebook_title.strip():
            payload = {
                "title": notebook_title.strip(),
                "owner_user_id": owner_user_id,
            }
            ok, result = api_post("/notebooks", payload)
            if ok:
                st.success("Notebook created.")
                st.json(result)
            else:
                st.error("Failed to create notebook.")
                st.code(str(result))
        else:
            st.error("Notebook title is required.")

    if st.button("Refresh notebooks"):
        st.rerun()

    ok, result = fetch_notebooks(owner_user_id)
    if ok:
        notebooks = result if isinstance(result, list) else []
        if notebooks:
            st.write("Your notebooks")
            st.dataframe(notebooks, use_container_width=True)

            notebook_options = {
                f"{n['id']} - {n['title']}": n
                for n in notebooks
                if isinstance(n, dict) and "id" in n and "title" in n
            }
            labels = list(notebook_options.keys())
            selected_label = st.selectbox("Select notebook to open", options=labels)
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

                source_tab, chat_tab = st.tabs(["Sources", "Chat"])

                with source_tab:
                    ok, notebook_result = fetch_notebooks(owner_user_id)
                    if not ok:
                        st.error("Failed to load notebooks.")
                        st.code(str(notebook_result))
                    else:
                        notebooks_for_owner = notebook_result if isinstance(notebook_result, list) else []
                        notebook_ids = {
                            n["id"]
                            for n in notebooks_for_owner
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

                                    form_data = {
                                        "owner_user_id": str(owner_user_id),
                                        "title": resolved_title,
                                        "status": source_status,
                                    }
                                    files_payload = {
                                        "file": (
                                            uploaded_file.name,
                                            uploaded_file.getvalue(),
                                            uploaded_file.type or "application/octet-stream",
                                        )
                                    }
                                    ok, create_result = api_post_multipart(
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
                                        "owner_user_id": owner_user_id,
                                        "type": source_type,
                                        "title": resolved_title,
                                        "original_name": resolved_original_name,
                                        "url": source_url or None,
                                        "storage_path": storage_path or None,
                                        "status": source_status,
                                    }
                                    ok, create_result = api_post(
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

                            ok, source_result = api_get(
                                f"/notebooks/{selected_notebook_id}/sources",
                                params={"owner_user_id": owner_user_id},
                            )
                            if ok:
                                sources = source_result if isinstance(source_result, list) else []
                                if sources:
                                    st.write("Sources")
                                    st.dataframe(sources, use_container_width=True)
                                else:
                                    st.info("No sources yet for this notebook.")
                            else:
                                st.error("Failed to fetch sources.")
                                st.code(str(source_result))

                with chat_tab:
                    st.write("Chat threads")
                    ok, thread_result = api_get(
                        f"/notebooks/{selected_notebook_id}/threads",
                        params={"owner_user_id": owner_user_id},
                    )
                    threads = thread_result if (ok and isinstance(thread_result, list)) else []

                    with st.form("create_thread_form"):
                        thread_title = st.text_input("Thread title (optional)")
                        create_thread_submitted = st.form_submit_button("Create thread")

                    if create_thread_submitted:
                        payload = {
                            "owner_user_id": owner_user_id,
                            "title": thread_title.strip() or None,
                        }
                        ok, create_thread_result = api_post(
                            f"/notebooks/{selected_notebook_id}/threads", payload
                        )
                        if ok:
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

                        ok, message_result = api_get(
                            f"/threads/{selected_thread_id}/messages",
                            params={"notebook_id": selected_notebook_id, "owner_user_id": owner_user_id},
                        )
                        if ok and isinstance(message_result, list):
                            st.write("Messages")
                            for msg in message_result:
                                role = msg.get("role", "unknown")
                                content = msg.get("content", "")
                                if role == "assistant":
                                    st.markdown(f"**Assistant:** {content}")
                                else:
                                    st.markdown(f"**You:** {content}")
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
                                    "owner_user_id": owner_user_id,
                                    "question": question.strip(),
                                    "top_k": 5,
                                }
                                ok, chat_result = api_post(
                                    f"/threads/{selected_thread_id}/chat?notebook_id={selected_notebook_id}",
                                    payload,
                                )
                                if ok:
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
        else:
            st.info("No notebooks yet for this user.")
    else:
        st.error("Failed to fetch notebooks.")
        st.code(str(result))
