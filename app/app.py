import os, re, asyncio
import nest_asyncio
import chromadb
import streamlit as st
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz

nest_asyncio.apply()

load_dotenv()
client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL  = "text-embedding-3-small"
GEN_MODEL    = "gpt-4o-mini"
CHAPTER      = "Chapter 10: Masked Language Models"
PDF_PATH = os.path.join(os.path.dirname(__file__), "chapter10.pdf")


def extract_and_clean(path):
    doc  = fitz.open(path)
    raw  = "\n".join(p.get_text() for p in doc)
    text = re.sub(r'^\s*\d+\s*$', '', raw, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()


def get_embedding(text):
    return client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding


async def enrich_chunk(chunk, document, title):
    prompt = f"""Title: {title}
{document[:4000]}
{chunk}

Provide brief context (1-2 sentences) explaining what this chunk discusses
in relation to the full document. Format: "This chunk from [title] discusses [explanation]." """
    resp = await async_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150)
    prefix = resp.choices[0].message.content.strip()
    return f"{prefix}\n\n{chunk}"


def build_col(chroma, name, chunk_list):
    try:
        chroma.delete_collection(name)
    except:
        pass
    col = chroma.create_collection(name)
    for i, chunk in enumerate(chunk_list):
        col.add(documents=[chunk], embeddings=[get_embedding(chunk)], ids=[f"c_{i}"])
    return col


@st.cache_resource(show_spinner=False)
def startup():
    full_text = extract_and_clean(PDF_PATH)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "])
    chunks = splitter.split_text(full_text)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    enriched = loop.run_until_complete(
        asyncio.gather(*[enrich_chunk(c, full_text, CHAPTER) for c in chunks])
    )

    chroma    = chromadb.Client()
    naive_col = build_col(chroma, "naive_rag_ch10",      chunks)
    ctx_col   = build_col(chroma, "contextual_rag_ch10", list(enriched))
    return naive_col, ctx_col


def query_rag(question, method, top_k=3):
    col     = ctx_col if method == "contextual" else naive_col
    q_emb   = get_embedding(question)
    results = col.query(query_embeddings=[q_emb], n_results=top_k)
    chunks  = results["documents"][0]
    context = "\n\n---\n\n".join(chunks)
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": (
                f"You are an expert assistant for '{CHAPTER}'. "
                "Answer ONLY from the provided context. Be concise and precise.")},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        temperature=0,
        max_tokens=400)
    return resp.choices[0].message.content.strip(), chunks


st.set_page_config(
    page_title="Chapter 10 · RAG QA",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background: #f7f6f2 !important; }
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, .stDeployButton { visibility: hidden !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
section[data-testid="stSidebar"] {
    display: flex !important; visibility: visible !important;
    width: 280px !important; min-width: 280px !important; max-width: 280px !important;
    transform: translateX(0) !important; background: #ffffff !important;
    border-right: 1px solid #e8e4dd !important;
    box-shadow: 2px 0 12px rgba(0,0,0,.04) !important;
}
[data-testid="collapsedControl"], button[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stChatMessage"] {
    background: #ffffff !important; border: 1px solid #ede9e2 !important;
    border-radius: 16px !important; padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 12px rgba(0,0,0,.04) !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important; border: 1.5px solid #ddd8cf !important;
    border-radius: 12px !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important; color: #1a1814 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #5b4ff5 !important;
    box-shadow: 0 0 0 3px rgba(91,79,245,.12) !important;
}
.stButton > button {
    background: #f7f6f2 !important; border: 1.5px solid #e0dbd2 !important;
    border-radius: 8px !important; color: #5a564f !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 12px !important; text-align: left !important;
}
.stButton > button:hover { background: #edeae4 !important; border-color: #c8c2b8 !important; }
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important; border: 1.5px solid #ddd8cf !important;
    border-radius: 8px !important; font-size: 13px !important;
}
hr { border-color: #e8e4dd !important; margin: 4px 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #ddd8cf; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

with st.spinner("⏳ Loading chapter and building vector index (first run only)…"):
    naive_col, ctx_col = startup()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px">
        <span style="font-size:26px">📖</span>
        <div>
            <div style="font-weight:700;font-size:15px;color:#1a1814;line-height:1.2">RAG Chat</div>
            <div style="font-size:10px;color:#9a9590;font-family:'JetBrains Mono',monospace">NLP · A6 · st126130</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9a9590;margin-bottom:6px">Method</p>', unsafe_allow_html=True)
    method = st.selectbox("method", label_visibility="collapsed",
        options=["contextual", "naive"],
        format_func=lambda x: "✨  Contextual Retrieval" if x == "contextual" else "⚡  Naive RAG")

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9a9590;margin:14px 0 6px">Top-K Chunks</p>', unsafe_allow_html=True)
    top_k = st.slider("topk", 1, 5, 3, label_visibility="collapsed")
    st.divider()

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9a9590;margin-bottom:8px">Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f7f6f2;border:1px solid #e8e4dd;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#9a9590;font-weight:700;letter-spacing:1px;text-transform:uppercase">Retriever</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5b4ff5;margin-bottom:8px">text-embedding-3-small</div>
        <div style="font-size:10px;color:#9a9590;font-weight:700;letter-spacing:1px;text-transform:uppercase">Generator</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#0ea5e9;margin-bottom:8px">gpt-4o-mini</div>
        <div style="font-size:10px;color:#9a9590;font-weight:700;letter-spacing:1px;text-transform:uppercase">Chapter</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9a9590">10 · Masked Language Models</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9a9590;margin-bottom:8px">💡 Try asking</p>', unsafe_allow_html=True)
    for s in [
        "What is masked language modeling?",
        "How does BERT training work?",
        "What is Next Sentence Prediction?",
        "What are contextual embeddings?",
        "How does NER use BIO tagging?"]:
        if st.button(s, use_container_width=True, key=f"sug_{s}"):
            st.session_state.prefill = s
            st.rerun()
    st.divider()

    if st.button("🗑  Clear chat", use_container_width=True, key="clear"):
        st.session_state.messages = []
        st.rerun()


badge = (
    '<span style="background:#d1fae518;color:#059669;border:1px solid #a7f3d0;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;font-family:\'JetBrains Mono\',monospace">✨ Contextual Retrieval</span>'
    if method == "contextual" else
    '<span style="background:#fef3c718;color:#d97706;border:1px solid #fde68a;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;font-family:\'JetBrains Mono\',monospace">⚡ Naive RAG</span>')

num_msgs   = len([m for m in st.session_state.messages if m["role"] == "user"])
meth_color = "#059669" if method == "contextual" else "#d97706"

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:12px 0 12px;border-bottom:1px solid #e8e4dd;margin-bottom:16px">
    <div>
        <h1 style="margin:0;font-size:20px;font-weight:700;color:#1a1814;line-height:1.2">Chapter 10 · QA System</h1>
        <p style="margin:2px 0 0;font-size:12px;color:#9a9590">Masked Language Models — Jurafsky &amp; Martin</p>
    </div>
    {badge}
</div>
<div style="display:flex;gap:12px;margin-bottom:20px">
    <div style="flex:1;background:#ffffff;border:1px solid #e8e4dd;border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
        <span style="font-size:24px">🤖</span>
        <div>
            <div style="font-size:11px;color:#9a9590;font-weight:700;text-transform:uppercase;letter-spacing:1px">Generator</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#1a1814;font-weight:500">gpt-4o-mini</div>
        </div>
    </div>
    <div style="flex:1;background:#ffffff;border:1px solid #e8e4dd;border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
        <span style="font-size:24px">🔍</span>
        <div>
            <div style="font-size:11px;color:#9a9590;font-weight:700;text-transform:uppercase;letter-spacing:1px">Retriever</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#1a1814;font-weight:500">text-embedding-3-small</div>
        </div>
    </div>
    <div style="flex:1;background:#ffffff;border:1px solid #e8e4dd;border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
        <span style="font-size:24px">💬</span>
        <div>
            <div style="font-size:11px;color:#9a9590;font-weight:700;text-transform:uppercase;letter-spacing:1px">Questions Asked</div>
            <div style="font-size:22px;color:#1a1814;font-weight:700;line-height:1">{num_msgs}</div>
        </div>
    </div>
    <div style="flex:1;background:#ffffff;border:1px solid #e8e4dd;border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
        <span style="font-size:24px">📚</span>
        <div>
            <div style="font-size:11px;color:#9a9590;font-weight:700;text-transform:uppercase;letter-spacing:1px">Chunks / Query</div>
            <div style="font-size:22px;color:{meth_color};font-weight:700;line-height:1">{top_k}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📖"):
        st.markdown(
            "👋 **Hello!** I'm your Chapter 10 assistant.\n\n"
            "Ask me anything about **BERT**, **Masked Language Modeling**, "
            "**Contextual Embeddings**, **NER**, **BIO Tagging**, and more.\n\n"
            "Pick your RAG method from the sidebar, then start asking!")

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "📖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Source chunks ({len(msg['sources'])} retrieved)"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"""
<div style="background:#faf9f6;border-left:3px solid #5b4ff5;border-radius:0 8px 8px 0;
            padding:10px 14px;margin-bottom:8px;font-family:'JetBrains Mono',monospace;
            font-size:11px;color:#6b6560;line-height:1.6;white-space:pre-wrap">
<strong style="color:#5b4ff5">Chunk {i+1}</strong><br><br>{chunk[:400]}{"…" if len(chunk)>400 else ""}
</div>""", unsafe_allow_html=True)

prefill = st.session_state.prefill
if prefill:
    st.session_state.prefill = ""

prompt = st.chat_input("Ask about BERT, MLM, Contextual Embeddings, NER…") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Retrieving and generating…"):
            try:
                answer, sources = query_rag(prompt, method, top_k)
            except Exception as e:
                answer, sources = f"⚠️ Error: {e}", []
        st.markdown(answer)
        if sources:
            with st.expander(f"📎 Source chunks ({len(sources)} retrieved)"):
                for i, chunk in enumerate(sources):
                    st.markdown(f"""
<div style="background:#faf9f6;border-left:3px solid #5b4ff5;border-radius:0 8px 8px 0;
            padding:10px 14px;margin-bottom:8px;font-family:'JetBrains Mono',monospace;
            font-size:11px;color:#6b6560;line-height:1.6;white-space:pre-wrap">
<strong style="color:#5b4ff5">Chunk {i+1}</strong><br><br>{chunk[:400]}{"…" if len(chunk)>400 else ""}
</div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant","content": answer,"sources": sources})
