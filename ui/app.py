"""
Streamlit UI — Live agent panel + HITL controls + RAGAS dashboard.
"""
import streamlit as st
import requests, json

API = "http://localhost:8000"

st.set_page_config(page_title="AI Research Assistant", page_icon="⬡", layout="wide")
st.markdown("""
<style>
.agent-step{padding:7px 14px;margin:3px 0;border-left:3px solid #00FF88;
background:#080E18;border-radius:4px;font-family:monospace;font-size:12px;color:#A0C0D8;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⬡ Research Assistant")
    st.divider()
    auto_search  = st.toggle("Auto-search arXiv", True)
    show_cot     = st.toggle("Show CoT Reasoning", True)
    show_sources = st.toggle("Show Sources", True)
    st.divider()
    st.subheader("Session History")
    for h in st.session_state.get("history", [])[-5:]:
        st.caption(f"• {h['query'][:50]}...")

col1, col2 = st.columns([3, 2])

with col1:
    st.header("Research Query")
    query   = st.text_area("Your question:", height=100,
        placeholder="What are the key limitations of RAG for multi-hop reasoning?")
    urls    = st.text_input("Paper URLs (comma-separated, optional):")
    run_btn = st.button("🔍 Start Research", type="primary", disabled=not query)

with col2:
    st.header("RAGAS Quality")
    if "scores" in st.session_state:
        for name, key, target in [
            ("Faithfulness",  "faithfulness",       0.85),
            ("Relevancy",     "answer_relevancy",   0.80),
            ("Precision",     "context_precision",  0.75),
            ("Recall",        "context_recall",     0.75),
        ]:
            score = st.session_state.scores.get(key, 0)
            icon  = "✅" if score >= target else "⚠️"
            st.progress(score, text=f"{name}: {score:.3f} {icon}")

if run_btn and query:
    st.divider()
    st.subheader("🤖 Live Agent Execution")

    paper_urls = [u.strip() for u in urls.split(",") if u.strip()] if urls else []
    resp = requests.post(f"{API}/research/start",
                         json={"query": query, "auto_search": auto_search, "paper_urls": paper_urls})
    data = resp.json()

    if data.get("from_cache"):
        st.success("⚡ Returned from semantic cache (< 100ms)")
        st.markdown(data["answer"])
        st.stop()

    st.session_state["session_id"] = data["session_id"]
    st.info(f"Session: {data['session_id']}")

    # HITL 1: review retrieved chunks
    st.subheader("📚 Review Retrieved Chunks")
    chunks = data.get("retrieved_chunks", [])
    keep   = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("meta", {})
        label = f"Chunk {i+1} [{meta.get('section','?')}] — {meta.get('paper_id','?')[:12]}"
        if st.checkbox(label, value=True, key=f"chunk_{i}"):
            keep.append(i)
        st.caption(chunk.get("text", "")[:280] + "...")

    feedback = st.text_input("Guidance for synthesis (optional):",
                             placeholder="Focus on 2024 results, ignore 2022 baseline")
    remove   = [i for i in range(len(chunks)) if i not in keep]

    if st.button("✅ Approve & Synthesize"):
        with st.spinner("Synthesizing..."):
            r2 = requests.post(f"{API}/research/approve_chunks", json={
                "session_id": st.session_state["session_id"],
                "remove_chunk_indices": remove,
                "feedback": feedback,
            })
            d2 = r2.json()

        st.subheader("🔍 Critic Review")
        st.markdown(f"**Faithfulness:** {d2.get('faithfulness_score', 0):.3f}")
        if d2.get("critique"):
            for issue in d2["critique"]:
                st.warning(f"⚠️ {issue}")
        else:
            st.success("✅ No issues found by critic")

        col_a, col_b = st.columns(2)
        with col_a:
            override = st.button("✅ Approve Answer")
        with col_b:
            revise   = st.button("🔄 Request Revision")

        if override or revise:
            with st.spinner("Finalizing..."):
                r3 = requests.post(f"{API}/research/resolve_critique", json={
                    "session_id": st.session_state["session_id"],
                    "override": override,
                })
                d3 = r3.json()

            st.divider()
            st.subheader("📋 Final Answer")
            st.markdown(d3.get("final_answer", ""))

            if d3.get("ragas_scores"):
                st.session_state["scores"] = d3["ragas_scores"]
                st.rerun()

            if show_sources and d3.get("sources"):
                with st.expander("📎 Sources"):
                    for s in d3["sources"]:
                        st.markdown(f"- {s}")

            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({"query": query, "answer": d3.get("final_answer", "")})
