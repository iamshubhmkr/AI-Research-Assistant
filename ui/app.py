"""Streamlit UI — query input, HITL chunk review, critique review (multi-round), RAGAS panel."""
import streamlit as st
import requests

API = "http://localhost:8000"
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant v3.1")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.stage = "ask"

if st.session_state.stage == "ask":
    query = st.text_input("Research question")
    urls = st.text_area("Optional document URLs (one per line)")
    auto = st.checkbox("Auto-search arXiv", value=True)
    if st.button("Start Research") and query:
        with st.spinner("Running fetch -> extract -> retrieve..."):
            resp = requests.post(f"{API}/research/start", json={
                "query": query, "auto_search": auto,
                "paper_urls": [u.strip() for u in urls.split("\n") if u.strip()]},
                timeout=600).json()
        if resp.get("from_cache"):
            st.success("Answer from cache")
            st.write(resp.get("answer", ""))
        elif resp.get("status") == "complete":
            st.warning(resp.get("note", "Pipeline finished early."))
            if resp.get("final_answer"):
                st.write(resp["final_answer"])
        else:
            st.session_state.session_id = resp["session_id"]
            st.session_state.chunks = resp["retrieved_chunks"]
            st.session_state.stage = "review_chunks"
            st.rerun()

elif st.session_state.stage == "review_chunks":
    st.subheader("HITL Checkpoint 1 — Review Retrieved Chunks")
    remove = []
    for i, c in enumerate(st.session_state.chunks):
        keep = st.checkbox(f"[{c.get('meta',{}).get('section','?')}] {c.get('text','')[:160]}...",
                           value=True, key=f"chunk{i}")
        if not keep:
            remove.append(i)
    feedback = st.text_input("Optional guidance for the synthesizer")
    if st.button("Approve & Synthesize"):
        with st.spinner("Synthesizing + fact-checking..."):
            resp = requests.post(f"{API}/research/approve_chunks", json={
                "session_id": st.session_state.session_id,
                "remove_chunk_indices": remove, "feedback": feedback},
                timeout=600).json()
        st.session_state.review = resp
        st.session_state.stage = "review_critique"
        st.rerun()

elif st.session_state.stage == "review_critique":
    st.subheader("HITL Checkpoint 2 — Review Critique")
    r = st.session_state.review
    st.caption(f"Revision round: {r.get('revision_count', 1)}")
    st.write(r.get("synthesis", ""))
    st.metric("Faithfulness", r.get("faithfulness_score", 0))
    if r.get("critique"):
        st.warning("Critic issues: " + "; ".join(r["critique"]))
    note = st.text_input("Optional revision guidance for the synthesizer")

    def _resolve(override):
        with st.spinner("Resolving..."):
            resp = requests.post(f"{API}/research/resolve_critique", json={
                "session_id": st.session_state.session_id, "override": override,
                "revision_note": note}, timeout=600).json()
        if resp.get("status") == "awaiting_critique_review":
            st.session_state.review = resp        # another round
        else:
            st.session_state.final = resp
            st.session_state.stage = "done"
        st.rerun()

    col1, col2 = st.columns(2)
    if col1.button("Accept / Revise per critic"):
        _resolve(False)
    if col2.button("Override Critic & Accept"):
        _resolve(True)

elif st.session_state.stage == "done":
    f = st.session_state.final
    st.subheader("Final Answer")
    st.write(f.get("final_answer", ""))
    st.caption("Sources: " + ", ".join(s for s in f.get("sources", []) if s))
    st.json({"cost_usd": f.get("estimated_cost_usd"), "tokens": f.get("token_usage")})
    if st.button("New Question"):
        st.session_state.stage = "ask"
        st.rerun()
