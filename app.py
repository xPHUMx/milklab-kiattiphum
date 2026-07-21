"""MilkLab° Gelato - Perfectly Aligned Modern Glass Cards Architecture.

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


# Gelato Catalog Data
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
        "rating": "4.9 ⭐ (128)",
        "desc": "นมสดฮอกไกโดนำเข้า 100% สกัดเข้มข้น รสสัมผัสนุ่มละมุน กลิ่นหอมกลมกล่อมเอกลักษณ์เฉพาะ MilkLab°",
        "tags": ["🥛 100% Hokkaido Milk", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "is_hero": True,
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
        "rating": "4.95 ⭐ (94)",
        "desc": "ดาร์กช็อกโกแลตพรีเมียมเข้มข้น 70% ให้รสสัมผัสขมนิดๆ หวานกำลังดี ไร้สารสังเคราะห์",
        "tags": ["🍫 Dark Cocoa 70%", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "is_hero": False,
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
        "rating": "4.88 ⭐ (112)",
        "desc": "สตรอว์เบอร์รีสด 100% รสชาติเปรี้ยวหวานสดชื่น ปลอดส่วนผสมของนม ปลอดไขมันเนย (Dairy-Free)",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥜 Nut-Free"],
        "is_hero": False,
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
        "rating": "4.92 ⭐ (86)",
        "desc": "ผงมัทฉะเกรดพิธีการสกัดจากเมืองอุจิ เกียวโต ให้ความหอมเข้มข้น มิติรสชาติลึกซึ้งแท้สไตล์ญี่ปุ่น",
        "tags": ["🍵 Uji Ceremonial", "🥜 Nut-Free", "👑 Premium Grade"],
        "is_hero": False,
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
        "rating": "4.85 ⭐ (74)",
        "desc": "เนื้อมะม่วงมหาชนกสดหวานฉ่ำคัดพิเศษ ปลอดนม 100% ให้ความสดชื่นคลายร้อนอย่างลงตัว",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥭 Fresh Mango"],
        "is_hero": False,
        "query": "ขอข้อมูลเจลาโต้มะม่วงมหาชนกซอร์เบต์ ราคาและส่วนผสม"
    }
]


# Dialog Assistant Modal
@st.dialog("💬 MilkLab° AI Assistant", width="large")
def open_ai_dialog(model, index, chunks, initial_query: str = ""):
    st.markdown("""
    <div style="background: rgba(255,255,255,0.7); backdrop-filter: blur(12px); border-radius: 16px; padding: 14px 20px; border: 1px solid rgba(255,255,255,0.9); margin-bottom: 15px;">
        <span style="font-size: 0.9rem; color: #334155;">
            🍨 <strong>MilkLab° RAG Concierge:</strong> สอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด หรือบริการจัดส่งได้ทันทีครับ
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
        if st.button("⏰ เวลาเปิด-ปิดร้าน", key="dlg_m1", use_container_width=True):
            quick_selected = "ร้านเปิดกี่โมงและปิดกี่โมง"
    with q_col2:
        if st.button("🚚 ค่าจัดส่ง & รัศมี", key="dlg_m2", use_container_width=True):
            quick_selected = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหนมีแพ็คเจลเย็นไหม"
    with q_col3:
        if st.button("🌱 เมนูสำหรับ Vegan", key="dlg_m3", use_container_width=True):
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
    # Streamlit Page Setup
    st.set_page_config(
        page_title="MilkLab° Gelato | Modern Glass Studio",
        page_icon="🍨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Ultra-Modern Liquid Glass & Perfectly Aligned Card Grid CSS Architecture
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'IBM Plex Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background Canvas */
    .stApp {
        background: radial-gradient(at 0% 0%, #FFF5F7 0px, transparent 50%),
                    radial-gradient(at 100% 0%, #F0F4FF 0px, transparent 50%),
                    radial-gradient(at 100% 100%, #F5F3FF 0px, transparent 50%),
                    radial-gradient(at 0% 100%, #FFFBEB 0px, transparent 50%);
        background-color: #FAFAFD;
        background-attachment: fixed;
    }
    
    /* Sleek Top Header Bar */
    .modern-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        padding: 18px 36px;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
        margin-bottom: 28px;
    }
    
    .nav-logo {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #0F172A;
    }
    
    .nav-pill {
        font-size: 0.75rem;
        background: rgba(225, 29, 72, 0.08);
        color: #E11D48;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: 1px solid rgba(225, 29, 72, 0.15);
    }
    
    .nav-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
    }
    
    .status-dot {
        width: 9px;
        height: 9px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
    }

    /* Hero Spotlight Card (Split Layout) */
    .hero-spotlight {
        background: linear-gradient(135deg, rgba(255,255,255,0.75) 0%, rgba(255,241,242,0.65) 100%);
        backdrop-filter: blur(24px) saturate(190%);
        -webkit-backdrop-filter: blur(24px);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        border-radius: 28px;
        padding: 36px 44px;
        box-shadow: 0 20px 40px rgba(225, 29, 72, 0.05);
        margin-bottom: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hero-tag {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #E11D48;
        margin-bottom: 8px;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.8px;
        margin-bottom: 10px;
    }
    
    .hero-desc {
        font-size: 1rem;
        color: #475569;
        line-height: 1.6;
        max-width: 600px;
        margin-bottom: 18px;
    }
    
    .hero-badge-group {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    
    .hero-badge {
        background: #FFFFFF;
        border: 1px solid #FFE4E8;
        color: #E11D48;
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    /* PERFECTLY EQUALIZED & ALIGNED CARD ARCHITECTURE */
    .modern-card-container {
        height: 380px;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    
    .modern-card-container:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.85);
        border-color: #FFE4E8;
        box-shadow: 0 20px 45px rgba(225, 29, 72, 0.1);
    }
    
    .card-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    
    .card-icon-frame {
        width: 64px;
        height: 64px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 1);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.03);
    }
    
    .card-price-container {
        text-align: right;
    }
    
    .card-price-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #E11D48;
    }
    
    .card-price-sub {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 500;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .card-subtitle {
        font-size: 0.8rem;
        color: #64748B;
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    .card-desc {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.55;
        height: 55px;
        overflow: hidden;
        margin-bottom: 12px;
    }
    
    .card-tags-group {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    
    .card-tag-pill {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #F1F5F9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.74rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
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
        background: linear-gradient(135deg, #FFFFFF 0%, #FFE4E8 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 12px 36px rgba(225, 29, 72, 0.28), 0 0 24px rgba(255, 255, 255, 0.9) !important;
        font-size: 2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        color: #E11D48 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.14) rotate(8deg) !important;
        box-shadow: 0 18px 45px rgba(225, 29, 72, 0.45) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load FAISS Index
    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"Error loading index: {exc}")
        st.stop()

    # Sleek Header Bar
    st.markdown("""
    <div class="modern-nav">
        <div class="nav-logo">
            🍨 MilkLab° <span style="font-weight: 400; color: #64748B;">Gelato</span>
            <span class="nav-pill">SYMMETRIC GRID UI</span>
        </div>
        <div class="nav-status">
            <span class="status-dot"></span>
            <span>Store Open (16:00 - 23:00) &nbsp;|&nbsp; 🚚 Express 5km Delivery</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Spotlight Card
    st.markdown("""
    <div class="hero-spotlight">
        <div>
            <div class="hero-tag">🔥 FLAVOR OF THE MONTH</div>
            <div class="hero-title">เจลาโต้นมสดฮอกไกโด 🍦</div>
            <div class="hero-desc">
                สัมผัสความหอมนุ่ม ละมุนลิ้น รสชาติสไตล์อาร์ติซานด้วยนมสดฮอกไกโดนำเข้า 100% สกัดเข้มข้น ไร้สารสังเคราะห์และปลอดถั่วทุกเมนู
            </div>
            <div class="hero-badge-group">
                <span class="hero-badge">⭐ 4.9 (128 รีวิว)</span>
                <span class="hero-badge">🥛 นมสดฮอกไกโด 100%</span>
                <span class="hero-badge">🥜 Nut-Free Guaranteed</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section Header & Category Filter
    st.markdown("## 🍨 Product Catalog Grid")
    st.caption("การ์ดเมนูจัดเรียงสวยงาม ความสูงสมดุลเท่ากันทุกช่อง คลิกปุ่มล่างการ์ดหรือคลิกปุ่มป็อบอัพทรงกลมมุมขวาล่างเพื่อคุยกับ AI")

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

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # PERFECTLY SYMMETRIC GRID RENDER (3 Columns Architecture)
    grid_cols = st.columns(3)

    for idx, item in enumerate(filtered_items):
        target_col = grid_cols[idx % 3]
        with target_col:
            tags_html = "".join(f'<span class="card-tag-pill">{t}</span>' for t in item["tags"])
            
            card_html = f"""
            <div class="modern-card-container">
                <div>
                    <div class="card-top-bar">
                        <div class="card-icon-frame">{item["icon"]}</div>
                        <div class="card-price-container">
                            <div class="card-price-num">{item["price"]} ฿</div>
                            <div class="card-price-sub">/{item["size"]} ({item["calories"]})</div>
                        </div>
                    </div>
                    <div class="card-title">{item["name"]}</div>
                    <div class="card-subtitle">{item["en_name"]} &nbsp;•&nbsp; {item["rating"]}</div>
                    <div class="card-desc">{item["desc"]}</div>
                </div>
                <div>
                    <div class="card-tags-group">{tags_html}</div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Action Button locked at uniform bottom baseline
            if st.button(f"✨ ถามน้อง AI เกี่ยวกับเมนูนี้", key=f"btn_ask_grid_{item['id']}", use_container_width=True):
                open_ai_dialog(model, index, chunks, initial_query=item["query"])

            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 📌 FIXED FLOATING CIRCULAR CHATBOT POPUP (BOTTOM-RIGHT)
    with st.popover("💬", help="คลิกเพื่อสอบถาม MilkLab° AI Assistant"):
        st.markdown("### 💬 MilkLab° AI Assistant")
        st.caption("สอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด และการจัดส่ง")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Quick Suggestions
        st.markdown("##### 💡 คำถามพบบ่อย:")
        p_c1, p_c2 = st.columns(2)
        quick_pop_q = None
        with p_c1:
            if st.button("⏰ เวลาเปิด-ปิดร้าน", key="p_g1", use_container_width=True):
                quick_pop_q = "ร้านเปิดกี่โมงและปิดกี่โมง"
        with p_c2:
            if st.button("🚚 ค่าส่ง & รัศมี", key="p_g2", use_container_width=True):
                quick_pop_q = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหน"

        # Chat Container
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

    # Studio Footer
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 20px 0;">
        MilkLab° Gelato • Symmetric Grid Architecture • Powered by Streamlit & Gemini RAG 🍨
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
