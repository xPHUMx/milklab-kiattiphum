"""MilkLab° Gelato - Liquid Glass (Glassmorphism) Minimal UI/UX & Floating Chatbot.

Run locally: streamlit run app.py
Deploy: push to GitHub then deploys to Streamlit Cloud / HuggingFace
"""

import os
import sys
import time
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@st.cache_resource
def load_index():
    """Load menu_kb.md, chunk text, encode with sentence-transformers, build FAISS index."""
    kb_path = os.path.join(os.path.dirname(__file__), "menu_kb.md")
    if not os.path.exists(kb_path):
        kb_path = "menu_kb.md"

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    current_section = ""

    for line in lines:
        if line.startswith("#"):
            current_section = line.replace("#", "").strip()
            continue
        
        if current_section:
            chunk_text = f"[{current_section}] {line.lstrip('- ').strip()}"
        else:
            chunk_text = line.lstrip('- ').strip()

        chunks.append(chunk_text)

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    return model, index, chunks


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 3) -> list[str]:
    """Encode query, search FAISS index, return top-k chunks."""
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, min(k, len(chunks)))

    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved.append(chunks[idx])
    return retrieved


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Send query + context to Gemini API with fallback handling."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ กรุณาตั้งค่า GOOGLE_API_KEY ในระบบก่อนใช้งาน AI Assistant ครับ"

    client = genai.Client(api_key=api_key)
    context_str = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""คุณเป็น AI ผู้ช่วยบริการลูกค้าของแบรนด์ไอศกรีมเจลาโต้ MilkLab° Gelato (ตอบเป็นภาษาไทยอย่างสุภาพ น่ารัก กระชับ เป็นกันเอง)
โปรดตอบคำถามโดยอ้างอิงจากข้อมูลบริบท (Context) ที่กำหนดให้ต่อไปนี้เท่านั้น
หากในข้อมูลบริบทไม่มีข้อมูลที่จะตอบคำถามได้ ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลดังกล่าวในระบบ"

บริบท (Context):
{context_str}

คำถามจากลูกค้า: {query}
"""

    models_to_try = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    last_error = None

    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as exc:
            last_error = exc
            time.sleep(1)
            continue

    return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {last_error}"


# Gelato Catalog Items
GELATO_CATALOG = [
    {
        "id": "hokkaido",
        "category": "Milk Series",
        "name": "เจลาโต้นมสดฮอกไกโด",
        "en_name": "Hokkaido Pure Milk Gelato",
        "icon": "🍦",
        "price": 80,
        "size": "120g",
        "calories": "195 kcal",
        "desc": "นมสดฮอกไกโดนำเข้า 100% สกัดเข้มข้น รสสัมผัสนุ่มละมุน กลิ่นหอมกลมกล่อมเอกลักษณ์เฉพาะ MilkLab°",
        "tags": ["🥛 100% Hokkaido Milk", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "is_signature": True,
        "query": "ขอรายละเอียดเจลาโต้นมสดฮอกไกโด สารแพ้อาหาร และราคา"
    },
    {
        "id": "chocolate",
        "category": "Milk Series",
        "name": "เจลาโต้ดาร์กช็อกโกแลต",
        "en_name": "Valrhona Dark Chocolate 70%",
        "icon": "🍫",
        "price": 85,
        "size": "120g",
        "calories": "210 kcal",
        "desc": "ดาร์กช็อกโกแลตพรีเมียมเข้มข้น 70% ให้รสสัมผัสขมนิดๆ หวานกำลังดี ไร้สารสังเคราะห์",
        "tags": ["🍫 Dark Cocoa 70%", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "is_signature": True,
        "query": "ขอรายละเอียดเจลาโต้ดาร์กช็อกโกแลต มีส่วนผสมอะไรบ้าง"
    },
    {
        "id": "strawberry",
        "category": "Sorbet (Vegan)",
        "name": "เจลาโต้สตรอว์เบอร์รีซอร์เบต์",
        "en_name": "Fresh Strawberry Sorbet",
        "icon": "🍓",
        "price": 85,
        "size": "120g",
        "calories": "110 kcal",
        "desc": "สตรอว์เบอร์รีสด 100% รสชาติเปรี้ยวหวานสดชื่น ปลอดส่วนผสมของนม ปลอดไขมันเนย (Dairy-Free)",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥜 Nut-Free"],
        "is_signature": False,
        "query": "คนทานมังสวิรัติหรือวีแกนกินเจลาโต้สตรอว์เบอร์รีซอร์เบต์ได้ไหม"
    },
    {
        "id": "matcha",
        "category": "Milk Series",
        "name": "เจลาโต้ชาเขียวมัทฉะ",
        "en_name": "Uji Ceremonial Matcha Gelato",
        "icon": "🍵",
        "price": 90,
        "size": "120g",
        "calories": "180 kcal",
        "desc": "ผงมัทฉะเกรดพิธีการสกัดจากเมืองอุจิ เกียวโต ให้ความหอมเข้มข้น มิติรสชาติลึกซึ้งแท้สไตล์ญี่ปุ่น",
        "tags": ["🍵 Uji Ceremonial", "🥜 Nut-Free", "👑 Premium"],
        "is_signature": True,
        "query": "เจลาโต้ชาเขียวมัทฉะราคาเท่าไหร่ ใช้วัตถุดิบจากไหน"
    },
    {
        "id": "mango",
        "category": "Sorbet (Vegan)",
        "name": "เจลาโต้มะม่วงมหาชนกซอร์เบต์",
        "en_name": "Mahachanok Mango Sorbet",
        "icon": "🥭",
        "price": 80,
        "size": "120g",
        "calories": "125 kcal",
        "desc": "เนื้อมะม่วงมหาชนกสดหวานฉ่ำคัดพิเศษ ปลอดนม 100% ให้ความสดชื่นคลายร้อนอย่างลงตัว",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥭 Fresh Mango"],
        "is_signature": False,
        "query": "ขอข้อมูลเจลาโต้มะม่วงมหาชนกซอร์เบต์ ราคาและส่วนผสม"
    }
]


# Dialog Modal for AI Chatbot (Accessible via Floating Circle or Cards)
@st.dialog("💬 MilkLab° AI Concierge (RAG Assistant)", width="large")
def open_chatbot_dialog(model, index, chunks, initial_query: str = ""):
    st.markdown("""
    <div style="background: rgba(255,255,255,0.6); backdrop-filter: blur(10px); border-radius: 14px; padding: 12px 18px; border: 1px solid rgba(255,255,255,0.8); margin-bottom: 12px;">
        <span style="font-size: 0.88rem; color: #475569;">
            🤖 <strong>AI Assistant Active:</strong> สอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด หรือบริการจัดส่งได้ทันทีครับ
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick Suggestion Pills
    st.caption("💡 คำถามยอดฮิตที่พบบ่อย:")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    quick_selected = None
    with q_col1:
        if st.button("⏰ เวลาเปิด-ปิดร้าน", key="float_q1", use_container_width=True):
            quick_selected = "ร้านเปิดกี่โมงและปิดกี่โมง"
    with q_col2:
        if st.button("🚚 ค่าจัดส่ง & รัศมี", key="float_q2", use_container_width=True):
            quick_selected = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหนมีแพ็คเจลเย็นไหม"
    with q_col3:
        if st.button("🌱 เมนูสำหรับ Vegan", key="float_q3", use_container_width=True):
            quick_selected = "มีเจลาโต้รสไหนบ้างที่คนทานมังสวิรัติหรือวีแกนกินได้"

    prompt_to_process = quick_selected or initial_query

    # Chat Display Container
    chat_box = st.container(height=340)
    with chat_box:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 30px; color: #94A3B8; font-size: 0.9rem;">
                👋 พิมพ์คำถามเกี่ยวกับเจลาโต้ด้านล่าง หรือคลิกเลือกปุ่มด้านบนเพื่อเริ่มต้นสนทนา
            </div>
            """, unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_text = st.chat_input("ถามอะไรเกี่ยวกับ MilkLab° Gelato...")
    final_prompt = user_text or prompt_to_process

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with chat_box:
            with st.chat_message("user"):
                st.write(final_prompt)

            with st.chat_message("assistant"):
                with st.spinner("กำลังค้นข้อมูลใน Knowledge Base..."):
                    context = retrieve_top_k(final_prompt, model, index, chunks)
                    answer = generate_answer(final_prompt, context)
                st.write(answer)
                with st.expander("🔍 ตรวจสอบบริบทข้อมูลอ้างอิง (Source Chunks)"):
                    for i, c in enumerate(context, 1):
                        st.markdown(f"**[{i}]** `{c}`")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    # Streamlit Page Config
    st.set_page_config(
        page_title="MilkLab° Gelato | Liquid Glass UI",
        page_icon="🍨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # State-of-the-Art Liquid Glass (Glassmorphism) CSS Design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'IBM Plex Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Liquid Gradient Backdrop */
    .stApp {
        background: radial-gradient(at 0% 0%, hsla(343,91%,93%,1) 0px, transparent 50%),
                    radial-gradient(at 100% 0%, hsla(242,100%,94%,1) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, hsla(170,100%,93%,1) 0px, transparent 50%),
                    radial-gradient(at 0% 100%, hsla(40,100%,94%,1) 0px, transparent 50%);
        background-color: #F8FAFC;
        background-attachment: fixed;
    }
    
    /* Liquid Glass Navbar Container */
    .glass-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 16px 32px;
        border-radius: 24px;
        box-shadow: 0 10px 30px 0 rgba(31, 38, 135, 0.05);
        margin-bottom: 24px;
    }
    
    .glass-brand {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #1E293B;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .glass-badge {
        font-size: 0.72rem;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        color: #E11D48;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(225, 29, 72, 0.1);
    }
    
    /* Liquid Glass Hero Header */
    .glass-hero {
        background: rgba(255, 255, 255, 0.38);
        backdrop-filter: blur(24px) saturate(190%);
        -webkit-backdrop-filter: blur(24px);
        border: 1.5px solid rgba(255, 255, 255, 0.7);
        border-radius: 28px;
        padding: 40px 48px;
        box-shadow: 0 16px 40px 0 rgba(31, 38, 135, 0.06);
        margin-bottom: 30px;
        text-align: center;
    }
    
    .glass-hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.8px;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #0F172A 0%, #E11D48 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .glass-hero-sub {
        font-size: 1.05rem;
        color: #475569;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Glass KPI Cards */
    .glass-kpi {
        background: rgba(255, 255, 255, 0.42);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.03);
        text-align: center;
    }
    
    .glass-kpi-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .glass-kpi-lbl {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Liquid Glass Product Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.48);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px);
        border: 1.5px solid rgba(255, 255, 255, 0.85);
        border-radius: 24px;
        padding: 26px;
        box-shadow: 0 12px 32px 0 rgba(31, 38, 135, 0.05);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        height: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-6px) scale(1.01);
        background: rgba(255, 255, 255, 0.65);
        border-color: rgba(255, 255, 255, 0.95);
        box-shadow: 0 20px 40px 0 rgba(225, 29, 72, 0.12);
    }
    
    .glass-card-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    
    .glass-icon-box {
        font-size: 2.6rem;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        width: 64px;
        height: 64px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    
    .glass-price {
        font-size: 1.35rem;
        font-weight: 800;
        color: #E11D48;
    }
    
    .glass-unit {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 400;
    }
    
    .glass-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    
    .glass-card-en {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 12px;
    }
    
    .glass-card-desc {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.55;
        margin-bottom: 16px;
    }
    
    .glass-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        color: #334155;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* 📌 FIXED FLOATING CIRCULAR CHATBOT POPUP (BOTTOM-RIGHT) */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        z-index: 999999 !important;
    }
    
    div[data-testid="stPopover"] > button {
        width: 68px !important;
        height: 68px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,228,232,0.95) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 2px solid rgba(255, 255, 255, 1) !important;
        box-shadow: 0 12px 36px rgba(225, 29, 72, 0.25), 0 0 20px rgba(255, 255, 255, 0.8) !important;
        font-size: 2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        color: #E11D48 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.12) rotate(8deg) !important;
        box-shadow: 0 16px 42px rgba(225, 29, 72, 0.4) !important;
        border-color: #FFE4E8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load FAISS Index & SentenceTransformer
    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"Error loading FAISS index: {exc}")
        st.stop()

    # Liquid Glass Navbar
    st.markdown("""
    <div class="glass-navbar">
        <div class="glass-brand">
            🍨 MilkLab° <span style="font-weight: 400; color: #475569;">Gelato</span>
            <span class="glass-badge">Liquid Glass UI</span>
        </div>
        <div style="font-size: 0.85rem; color: #475569; font-weight: 500;">
            🟢 Open Daily (16:00 - 23:00) &nbsp;|&nbsp; 🚚 5km Express Delivery
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Liquid Glass Hero Header
    st.markdown("""
    <div class="glass-hero">
        <div class="glass-hero-title">Artisan Homemade Gelato</div>
        <div class="glass-hero-sub">
            สัมผัสความนุ่ม ละมุน รสชาติพรีเมียมเข้มข้นสไตล์อาร์ติซาน ปลอดสารสังเคราะห์ 100% สอบถามข้อมูลสินค้า สารแพ้อาหาร หรือบริการจัดส่งผ่านระบบ AI Vector Search ได้ทันที
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Glass KPI Banner
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="glass-kpi">
            <div class="glass-kpi-val">5 Flavors</div>
            <div class="glass-kpi-lbl">Signature Menus</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="glass-kpi">
            <div class="glass-kpi-val">100% Nut-Free</div>
            <div class="glass-kpi-lbl">Safe Guaranteed</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="glass-kpi">
            <div class="glass-kpi-val">5 km Radius</div>
            <div class="glass-kpi-lbl">Cold Pack Delivery</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="glass-kpi">
            <div class="glass-kpi-val">Vegan Options</div>
            <div class="glass-kpi-lbl">Fresh Sorbet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # Header Title & Filter Selection
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown("## 🍧 Gelato Menu Showcase")
        st.caption("เลือกดูเมนูไอศกรีมเจลาโต้ หรือคลิกปุ่มป็อบอัพแชทบอททรงกลมที่มุมขวาล่างเพื่อคุยกับน้อง AI")

    selected_category = st.radio(
        "หมวดหมู่สินค้า:",
        ["ทั้งหมด (All Flavors)", "Milk Series (สูตรนมสด)", "Sorbet (Vegan / ปลอดนม)"],
        horizontal=True
    )

    # Filter Items
    filtered_items = GELATO_CATALOG
    if "Milk Series" in selected_category:
        filtered_items = [x for x in GELATO_CATALOG if x["category"] == "Milk Series"]
    elif "Sorbet" in selected_category:
        filtered_items = [x for x in GELATO_CATALOG if x["category"] == "Sorbet (Vegan)"]

    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    # Gelato Cards Grid (3 Columns)
    grid_cols = st.columns(3)

    for idx, item in enumerate(filtered_items):
        target_col = grid_cols[idx % 3]
        with target_col:
            tags_html = "".join(f'<span class="glass-pill">{t}</span>' for t in item["tags"])
            
            card_html = f"""
            <div class="glass-card">
                <div>
                    <div class="glass-card-head">
                        <div class="glass-icon-box">{item["icon"]}</div>
                        <div style="text-align: right;">
                            <div class="glass-price">{item["price"]} ฿</div>
                            <div class="glass-unit">/{item["size"]} ({item["calories"]})</div>
                        </div>
                    </div>
                    <div class="glass-card-title">{item["name"]}</div>
                    <div class="glass-card-en">{item["en_name"]}</div>
                    <div style="margin-bottom: 10px;">{tags_html}</div>
                    <div class="glass-card-desc">{item["desc"]}</div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Button for card inquiry
            if st.button(f"🔍 สอบถามเมนูนี้", key=f"btn_ask_glass_{item['id']}", use_container_width=True):
                open_chatbot_dialog(model, index, chunks, initial_query=item["query"])

            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 📌 FIXED FLOATING CIRCULAR CHATBOT POPUP AT BOTTOM-RIGHT
    with st.popover("💬", help="คลิกเพื่อสอบถาม MilkLab° AI Concierge"):
        st.markdown("### 💬 MilkLab° AI Concierge")
        st.caption("สอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด และการจัดส่ง")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Quick Suggestion Buttons
        st.markdown("##### 💡 คำถามพบบ่อย:")
        p_c1, p_c2 = st.columns(2)
        quick_pop_q = None
        with p_c1:
            if st.button("⏰ เวลาเปิด-ปิดร้าน", key="pop_q1", use_container_width=True):
                quick_pop_q = "ร้านเปิดกี่โมงและปิดกี่โมง"
        with p_c2:
            if st.button("🚚 ค่าส่ง & รัศมี", key="pop_q2", use_container_width=True):
                quick_pop_q = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหน"

        # Chat History Container
        pop_chat_box = st.container(height=300)
        with pop_chat_box:
            if not st.session_state.messages:
                st.info("👋 สวัสดีครับ! น้อง AI ยินดีให้บริการ สอบถามข้อมูล MilkLab° Gelato ได้เลยครับ 😊")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        pop_input = st.chat_input("พิมพ์คำถามเจลาโต้ที่นี่...")
        prompt_run = pop_input or quick_pop_q

        if prompt_run:
            st.session_state.messages.append({"role": "user", "content": prompt_run})
            with pop_chat_box:
                with st.chat_message("user"):
                    st.write(prompt_run)

                with st.chat_message("assistant"):
                    with st.spinner("กำลังค้นข้อมูล..."):
                        context = retrieve_top_k(prompt_run, model, index, chunks)
                        answer = generate_answer(prompt_run, context)
                    st.write(answer)
                    with st.expander("🔍 ข้อมูลอ้างอิง (Source Chunks)"):
                        for i, c in enumerate(context, 1):
                            st.markdown(f"**[{i}]** `{c}`")

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    st.markdown("---")

    # Liquid Glass Footer
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 20px 0;">
        MilkLab° Gelato • Minimal Liquid Glass UI/UX Design • Powered by Streamlit & Gemini RAG 🍨
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
