"""MilkLab° Gelato - High-Fashion DTC Poster & Flavor Carousel Experience.
Reference Style: HANCI Floating Pill Top Bar / Full Moon Party / Karaoke Night

Run locally: streamlit run app.py
Deploy: push to GitHub then deploys to Streamlit Cloud / HuggingFace
"""

import os
import sys
import time
import textwrap
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


# Gelato Catalog with Exact Reference Poster Aesthetics (Full Moon Party / Karaoke Night Style)
GELATO_POSTERS = [
    {
        "id": "mango",
        "top_tag": "MANGO CHAMOY",
        "sub_header": "IF THAILAND'S FULL MOON PARTY WAS A FLAVOR",
        "title_part1": "FULL MOON",
        "title_part2": "PARTY",
        "th_name": "เจลาโต้มะม่วงมหาชนกซอร์เบต์",
        "tub_title": "MANGO × CHAMOY",
        "footer_tag": "IT'S A PARTY FOR YOUR TASTE BUDS! 100% VEGAN FRESH SORBET",
        "price": 80,
        "size": "120g",
        "bg_color": "#FFDF00",          # Vibrant Yellow
        "text_color": "#D80000",        # Bold Red Typography
        "sub_text_color": "#D80000",
        "tub_bg": "#FFCC00",
        "photo_url": "https://images.unsplash.com/photo-1553279768-865429fa0078?q=80&w=800&auto=format&fit=crop",
        "left_preview": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?q=80&w=300&auto=format&fit=crop",
        "right_preview": "https://images.unsplash.com/photo-1511381939415-e44015466834?q=80&w=300&auto=format&fit=crop",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥭 Fresh Mango"],
        "query": "ขอข้อมูลเจลาโต้มะม่วงมหาชนกซอร์เบต์ ราคาและส่วนผสม"
    },
    {
        "id": "strawberry",
        "top_tag": "VELVETY STRAWBERRY SHORTCAKE",
        "sub_header": "JUST THE THING TO MAKE YOU SING!",
        "title_part1": "KARAOKE",
        "title_part2": "NIGHT",
        "th_name": "เจลาโต้สตรอว์เบอร์รีซอร์เบต์",
        "tub_title": "STRAWBERRY × SHORTCAKE",
        "footer_tag": "BRILLIANT RED STRAWBERRY JAM • SWEET & TANGY DELIGHT",
        "price": 85,
        "size": "120g",
        "bg_color": "#FFB7C5",          # Romantic Soft Pink
        "text_color": "#C1121F",        # Deep Red Typography
        "sub_text_color": "#C1121F",
        "tub_bg": "#FFA4B6",
        "photo_url": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?q=80&w=800&auto=format&fit=crop",
        "left_preview": "https://images.unsplash.com/photo-1553279768-865429fa0078?q=80&w=300&auto=format&fit=crop",
        "right_preview": "https://images.unsplash.com/photo-1550583724-b2692b85b150?q=80&w=300&auto=format&fit=crop",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥜 Nut-Free"],
        "query": "ขอข้อมูลเจลาโต้สตรอว์เบอร์รีซอร์เบต์ รสชาติ สารแพ้อาหาร และราคา"
    },
    {
        "id": "hokkaido",
        "top_tag": "ORGANIC HOKKAIDO FRESH MILK",
        "sub_header": "CREAMY RICHNESS DIRECT FROM HOKKAIDO",
        "title_part1": "HOKKAIDO",
        "title_part2": "DREAM",
        "th_name": "เจลาโต้นมสดฮอกไกโด",
        "tub_title": "HOKKAIDO × PURE MILK",
        "footer_tag": "100% IMPORTED HOKKAIDO MILK • SIGNATURE #1 FLAVOR",
        "price": 80,
        "size": "120g",
        "bg_color": "#FFEAD2",          # Creamy Milk Beige
        "text_color": "#B45309",        # Warm Terracotta Typography
        "sub_text_color": "#B45309",
        "tub_bg": "#FCD34D",
        "photo_url": "https://images.unsplash.com/photo-1550583724-b2692b85b150?q=80&w=800&auto=format&fit=crop",
        "left_preview": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?q=80&w=300&auto=format&fit=crop",
        "right_preview": "https://images.unsplash.com/photo-1576092768241-dec231879fc3?q=80&w=300&auto=format&fit=crop",
        "tags": ["🥛 100% Hokkaido Milk", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "query": "ขอรายละเอียดเจลาโต้นมสดฮอกไกโด สารแพ้อาหาร และราคา"
    },
    {
        "id": "chocolate",
        "top_tag": "VALRHONA DARK COCOA 70%",
        "sub_header": "LUSCIOUS DEEP CHOCOLATE EXPERIENCE",
        "title_part1": "MIDNIGHT",
        "title_part2": "COCOA",
        "th_name": "เจลาโต้ดาร์กช็อกโกแลต",
        "tub_title": "DARK × CHOCOLATE",
        "footer_tag": "70% PREMIUM DARK CHOCOLATE • INTENSE & BITTERSWEET",
        "price": 85,
        "size": "120g",
        "bg_color": "#D8C4B6",          # Rich Cocoa Dust
        "text_color": "#3E2723",        # Deep Chocolate Typography
        "sub_text_color": "#3E2723",
        "tub_bg": "#6D4C41",
        "photo_url": "https://images.unsplash.com/photo-1511381939415-e44015466834?q=80&w=800&auto=format&fit=crop",
        "left_preview": "https://images.unsplash.com/photo-1550583724-b2692b85b150?q=80&w=300&auto=format&fit=crop",
        "right_preview": "https://images.unsplash.com/photo-1576092768241-dec231879fc3?q=80&w=300&auto=format&fit=crop",
        "tags": ["🍫 Dark Cocoa 70%", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "query": "ขอรายละเอียดเจลาโต้ดาร์กช็อกโกแลต มีส่วนผสมอะไรบ้าง"
    },
    {
        "id": "matcha",
        "top_tag": "UJI CEREMONIAL MATCHA",
        "sub_header": "TRADITIONAL UJI JAPANESE TEA CEREMONY",
        "title_part1": "KYOTO",
        "title_part2": "CEREMONY",
        "th_name": "เจลาโต้ชาเขียวมัทฉะ",
        "tub_title": "UJI × MATCHA",
        "footer_tag": "CEREMONIAL GRADE MATCHA FROM UJI, KYOTO • RICH UMAMI",
        "price": 90,
        "size": "120g",
        "bg_color": "#C2E2B4",          # Japanese Green Tea
        "text_color": "#1E4D2B",        # Forest Green Typography
        "sub_text_color": "#1E4D2B",
        "tub_bg": "#4D7C0F",
        "photo_url": "https://images.unsplash.com/photo-1576092768241-dec231879fc3?q=80&w=800&auto=format&fit=crop",
        "left_preview": "https://images.unsplash.com/photo-1511381939415-e44015466834?q=80&w=300&auto=format&fit=crop",
        "right_preview": "https://images.unsplash.com/photo-1553279768-865429fa0078?q=80&w=300&auto=format&fit=crop",
        "tags": ["🍵 Uji Ceremonial", "🥜 Nut-Free", "👑 Premium Grade"],
        "query": "เจลาโต้ชาเขียวมัทฉะราคาเท่าไหร่ ใช้วัตถุดิบจากไหน"
    }
]


# Dialog Assistant Modal
@st.dialog("💬 MilkLab° AI Concierge", width="large")
def open_ai_dialog(model, index, chunks, initial_query: str = ""):
    st.markdown("""
    <div style="background: rgba(255,255,255,0.7); backdrop-filter: blur(12px); border-radius: 16px; padding: 14px 20px; border: 1px solid rgba(255,255,255,0.9); margin-bottom: 15px;">
        <span style="font-size: 0.9rem; color: #334155;">
            🍨 <strong>MilkLab° RAG Assistant:</strong> สอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด หรือบริการจัดส่งได้ทันทีครับ
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick Suggestion Buttons
    st.caption("💡 คำถามพบบ่อย:")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    quick_selected = None
    with q_col1:
        if st.button("⏰ เวลาเปิด-ปิดร้าน", key="dlg_p1", use_container_width=True):
            quick_selected = "ร้านเปิดกี่โมงและปิดกี่โมง"
    with q_col2:
        if st.button("🚚 ค่าจัดส่ง & รัศมี", key="dlg_p2", use_container_width=True):
            quick_selected = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหนมีแพ็คเจลเย็นไหม"
    with q_col3:
        if st.button("🌱 เมนูสำหรับ Vegan", key="dlg_p3", use_container_width=True):
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

    user_text = st.chat_input("สอบถามข้อมูล MilkLab° Gelato...")
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
                with st.expander("🔍 ข้อมูลอ้างอิง (Source Chunks)"):
                    for i, c in enumerate(context, 1):
                        st.markdown(f"**[{i}]** `{c}`")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    st.set_page_config(
        page_title="MILKLAB × GELATO | High-Fashion Experience",
        page_icon="🍨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    if "poster_idx" not in st.session_state:
        st.session_state.poster_idx = 0

    curr = GELATO_POSTERS[st.session_state.poster_idx]

    # High-Fashion Header & Poster CSS Architecture (Exact HANCI Floating Pill Navbar Reference!)
    css_code = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Kanit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Kanit', 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Background Dynamic Color Shift */
    .stApp {{
        background-color: {curr["bg_color"]} !important;
        transition: background-color 0.7s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    /* 🌟 EXACT HANCI FLOATING PILL TOP NAVBAR REFERENCE ARCHITECTURE */
    .hanci-topbar-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        padding: 10px 0 20px 0;
    }}
    
    .hanci-topbar-pill {{
        background: #FFFFFF;
        border-radius: 50px;
        padding: 12px 36px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 95%;
        max-width: 1100px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }}
    
    .topbar-left-menu, .topbar-right-menu {{
        display: flex;
        align-items: center;
        gap: 22px;
    }}
    
    .topbar-link {{
        font-family: 'Kanit', sans-serif;
        font-size: 0.92rem;
        font-weight: 600;
        color: #1E293B;
        text-decoration: none;
        transition: color 0.2s ease;
    }}
    
    .topbar-link:hover {{
        color: #E11D48;
    }}
    
    .topbar-center-logo {{
        font-family: 'Anton', sans-serif;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 2px;
        color: #0F172A;
    }}
    
    .topbar-icons {{
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 1.1rem;
        color: #1E293B;
    }}
    
    .topbar-icon-cart {{
        position: relative;
        cursor: pointer;
    }}
    
    .cart-count-badge {{
        position: absolute;
        top: -6px;
        right: -8px;
        background: #E11D48;
        color: #FFFFFF;
        font-size: 0.65rem;
        font-weight: 800;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    /* Exact Reference Poster Canvas */
    .poster-canvas {{
        position: relative;
        text-align: center;
        padding: 15px 20px 35px 20px;
        min-height: 560px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        overflow: hidden;
    }}
    
    .poster-top-tag {{
        font-family: 'Anton', 'Kanit', sans-serif;
        font-size: 1.6rem;
        letter-spacing: 2px;
        color: {curr["text_color"]};
        text-transform: uppercase;
        margin-bottom: 2px;
    }}
    
    .poster-sub-header {{
        font-family: 'Kanit', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: {curr["text_color"]};
        max-width: 600px;
        margin: 0 auto 15px auto;
        opacity: 0.9;
    }}
    
    /* GIANT BACKGROUND DISPLAY TYPOGRAPHY */
    .poster-giant-text {{
        font-family: 'Anton', sans-serif;
        font-size: 7.8rem;
        font-weight: 900;
        line-height: 0.88;
        letter-spacing: 2px;
        color: {curr["text_color"]};
        text-transform: uppercase;
        user-select: none;
        z-index: 1;
    }}
    
    /* CENTRAL RECTANGULAR PHOTO FRAME WITH OVERLAPPING TUB */
    .poster-center-box {{
        position: relative;
        width: 520px;
        height: 220px;
        margin: -100px auto -80px auto;
        z-index: 2;
    }}
    
    .poster-photo-rect {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 4px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.18);
    }}
    
    /* OVERLAPPING BRANDED ICE CREAM TUB IN EXACT CENTER */
    .poster-tub-center {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 170px;
        height: 170px;
        background: {curr["tub_bg"]};
        border-radius: 20px 20px 45px 45px;
        border: 4px solid #FFFFFF;
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 12px;
        animation: floatTub 4s ease-in-out infinite;
        z-index: 3;
    }}
    
    .tub-lid {{
        position: absolute;
        top: -12px;
        width: 180px;
        height: 22px;
        background: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    
    .tub-label-brand {{
        font-family: 'Anton', sans-serif;
        font-size: 1.45rem;
        line-height: 1;
        color: {curr["text_color"]};
        text-transform: uppercase;
        margin-top: 8px;
    }}
    
    .poster-footer-text {{
        font-family: 'Kanit', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: {curr["text_color"]};
        margin-top: 20px;
        z-index: 2;
    }}
    
    /* SIDE CAROUSEL CUTOUT PREVIEWS */
    .side-cutout-left {{
        position: absolute;
        left: -40px;
        top: 35%;
        width: 90px;
        height: 180px;
        border-radius: 0 16px 16px 0;
        overflow: hidden;
        box-shadow: 5px 10px 25px rgba(0,0,0,0.15);
        opacity: 0.85;
    }}
    
    .side-cutout-right {{
        position: absolute;
        right: -40px;
        top: 35%;
        width: 90px;
        height: 180px;
        border-radius: 16px 0 0 16px;
        overflow: hidden;
        box-shadow: -5px 10px 25px rgba(0,0,0,0.15);
        opacity: 0.85;
    }}

    @keyframes floatTub {{
        0%, 100% {{ transform: translate(-50%, -50%) translateY(0px); }}
        50% {{ transform: translate(-50%, -50%) translateY(-10px); }}
    }}

    /* 📌 FIXED FLOATING CIRCULAR CHATBOT POPUP (BOTTOM-RIGHT) */
    div[data-testid="stPopover"] {{
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        z-index: 999999 !important;
    }}
    
    div[data-testid="stPopover"] > button {{
        width: 72px !important;
        height: 72px !important;
        border-radius: 50% !important;
        background: #FFFFFF !important;
        border: 3px solid {curr["text_color"]} !important;
        box-shadow: 0 12px 36px rgba(0,0,0,0.25) !important;
        font-size: 2.2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        color: {curr["text_color"]} !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stPopover"] > button:hover {{
        transform: scale(1.15) rotate(8deg) !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.35) !important;
    }}
    </style>
    """
    st.markdown(textwrap.dedent(css_code), unsafe_allow_html=True)

    # Load FAISS Index
    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"Error loading index: {exc}")
        st.stop()

    # 🌟 EXACT HANCI FLOATING PILL TOP BAR (DIRECT REFERENCE MATCH)
    hanci_topbar_html = """<div class="hanci-topbar-wrapper">
<div class="hanci-topbar-pill">
<div class="topbar-left-menu">
<a href="#shop" class="topbar-link">Shop</a>
<a href="#about" class="topbar-link">About us</a>
<a href="#flavours" class="topbar-link">Flavours ▾</a>
<a href="#learn" class="topbar-link">Learn</a>
</div>
<div class="topbar-center-logo">MILKLAB</div>
<div class="topbar-right-menu">
<a href="#story" class="topbar-link">Our Story</a>
<a href="#contact" class="topbar-link">Contact</a>
<div class="topbar-icons">
<span class="topbar-icon" title="Search">🔍</span>
<span class="topbar-icon" title="Wishlist">♡</span>
<span class="topbar-icon-cart" title="Cart">🛍️ <span class="cart-count-badge">0</span></span>
</div>
</div>
</div>
</div>"""
    st.markdown(hanci_topbar_html, unsafe_allow_html=True)

    # Interactive Flavor Switcher Navigation (DTC Style)
    st.markdown("### 🍨 เลือกชมโปสเตอร์รสชาติไอศกรีม (Click to Change Poster Theme):")
    nav_cols = st.columns(len(GELATO_POSTERS))

    for idx, p in enumerate(GELATO_POSTERS):
        with nav_cols[idx]:
            is_active = (idx == st.session_state.poster_idx)
            label = f"✨ {p['top_tag']}" if is_active else p['top_tag']
            if st.button(label, key=f"nav_p_{p['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.poster_idx = idx
                st.rerun()

    # EXACT REFERENCE POSTER CONTAINER
    poster_html = f"""<div class="poster-canvas">
<div class="side-cutout-left"><img src="{curr['left_preview']}" style="width:100%; height:100%; object-fit:cover;" /></div>
<div class="side-cutout-right"><img src="{curr['right_preview']}" style="width:100%; height:100%; object-fit:cover;" /></div>
<div>
<div class="poster-top-tag">{curr["top_tag"]}</div>
<div class="poster-sub-header">{curr["sub_header"]}</div>
<div class="poster-giant-text">{curr["title_part1"]}</div>
</div>
<div class="poster-center-box">
<img src="{curr['photo_url']}" class="poster-photo-rect" alt="Fruit Photo" />
<div class="poster-tub-center">
<div class="tub-lid"></div>
<div style="font-size: 0.75rem; font-weight: 800; color: #FFFFFF; letter-spacing: 1px;">MILKLAB°</div>
<div class="tub-label-brand">{curr["tub_title"]}</div>
<div style="font-size: 0.65rem; color: #FFFFFF; margin-top: 4px;">NET WT 120G</div>
</div>
</div>
<div>
<div class="poster-giant-text">{curr["title_part2"]}</div>
<div class="poster-footer-text">{curr["footer_tag"]} &nbsp;•&nbsp; PRICE: {curr["price"]} THB</div>
</div>
</div>"""

    st.markdown(poster_html, unsafe_allow_html=True)

    # Action Bar for Poster
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(f"✨ สอบถามข้อมูลน้อง AI เกี่ยวกับ {curr['th_name']}", key="btn_poster_ask", type="primary", use_container_width=True):
            open_ai_dialog(model, index, chunks, initial_query=curr["query"])

    st.markdown("---")

    # Product Cards Grid (Matching Reference Image 2!)
    st.markdown("## 🎴 ALL FLAVOR POSTER CARDS (คลิกเลือกดูโปสเตอร์ทุกรสชาติ)")
    grid_cols = st.columns(3)

    for idx, item in enumerate(GELATO_POSTERS):
        target_col = grid_cols[idx % 3]
        with target_col:
            tags_html = "".join(f'<span style="background:rgba(255,255,255,0.7); color:{item["text_color"]}; padding:4px 10px; border-radius:10px; font-size:0.78rem; font-weight:700; margin-right:4px;">{t}</span>' for t in item["tags"])
            
            card_html = f"""<div style="background: {item['bg_color']}; border-radius: 24px; padding: 24px; text-align: center; box-shadow: 0 12px 30px rgba(0,0,0,0.08); margin-bottom: 15px; border: 2px solid rgba(255,255,255,0.8);">
<div style="height: 120px; border-radius: 12px; overflow: hidden; margin-bottom: 14px;">
<img src="{item['photo_url']}" style="width: 100%; height: 100%; object-fit: cover;" />
</div>
<div style="font-family: 'Anton', sans-serif; font-size: 1.8rem; color: {item['text_color']}; line-height: 1;">{item['title_part1']} {item['title_part2']}</div>
<div style="font-weight: 700; color: {item['text_color']}; font-size: 0.9rem; margin-top: 4px; margin-bottom: 10px;">{item['th_name']}</div>
<div style="margin-bottom: 12px;">{tags_html}</div>
<div style="font-size: 1.4rem; font-weight: 900; color: {item['text_color']};">{item['price']} THB <span style="font-size: 0.85rem; font-weight: 500;">/ {item['size']}</span></div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button(f"✨ ดูโปสเตอร์รสนี้ ({item['th_name']})", key=f"btn_card_select_{item['id']}", use_container_width=True):
                st.session_state.poster_idx = idx
                st.rerun()

            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 📌 FIXED FLOATING CIRCULAR CHATBOT POPUP (BOTTOM-RIGHT)
    with st.popover("💬", help="คลิกเพื่อสอบถาม MilkLab° AI Assistant"):
        st.markdown("### 💬 MilkLab° AI Assistant")
        st.caption("สอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด และการจัดส่ง")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.markdown("##### 💡 คำถามพบบ่อย:")
        p_c1, p_c2 = st.columns(2)
        quick_pop_q = None
        with p_c1:
            if st.button("⏰ เวลาเปิด-ปิดร้าน", key="p_post1", use_container_width=True):
                quick_pop_q = "ร้านเปิดกี่โมงและปิดกี่โมง"
        with p_c2:
            if st.button("🚚 ค่าส่ง & รัศมี", key="p_post2", use_container_width=True):
                quick_pop_q = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหน"

        pop_chat_box = st.container(height=300)
        with pop_chat_box:
            if not st.session_state.messages:
                st.info("👋 สวัสดีครับ! น้อง AI ยินดีให้บริการ สอบถามข้อมูล MilkLab° Gelato ได้เลยครับ 😊")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        pop_input = st.chat_input("ถามอะไรเกี่ยวกับ MilkLab° Gelato...")
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

    st.markdown("""<div style="text-align: center; color: #475569; font-size: 0.85rem; padding: 20px 0; font-weight: 700;">
MILKLAB × GELATO • HIGH-FASHION POSTER SHOWCASE • GEMINI RAG AI 🍨
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
