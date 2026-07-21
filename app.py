"""MilkLab° Gelato - Professional Commercial E-Commerce & AI Assistant.

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

    prompt = f"""คุณเป็น AI Concierge มืออาชีพของแบรนด์ไอศกรีมเจลาโต้ MilkLab° Gelato (ตอบเป็นภาษาไทยอย่างสุภาพ กระชับ และเป็นกันเอง)
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
        "desc": "นมสดฮอกไกโดนำเข้า 100% สกัดเข้มข้น รสสัมผัสนุ่มละมุน กลิ่นหอมกลมกล่อมเอกลักษณ์เฉพาะ MilkLab°",
        "tags": ["🥛 100% Hokkaido Milk", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "is_bestseller": True,
        "query": "ขอข้อมูลเจลาโต้นมสดฮอกไกโด รสชาติ สารแพ้อาหาร และราคา"
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
        "tags": ["🍫 Dark Chocolate 70%", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "is_bestseller": True,
        "query": "ขอข้อมูลเจลาโต้ดาร์กช็อกโกแลต มีส่วนผสมอะไรบ้าง"
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
        "is_bestseller": False,
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
        "tags": ["🍵 Uji Ceremonial Grade", "🥜 Nut-Free", "👑 Premium"],
        "is_bestseller": True,
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
        "is_bestseller": False,
        "query": "ขอข้อมูลเจลาโต้มะม่วงมหาชนกซอร์เบต์ ราคาและส่วนผสม"
    }
]


# AI Assistant Dialog Popup
@st.dialog("💬 MilkLab° AI Concierge", width="large")
def open_ai_concierge_dialog(model, index, chunks, initial_query: str = ""):
    st.markdown("""
    <div style="background: #F8FAFC; border-radius: 12px; padding: 14px 18px; border: 1px solid #E2E8F0; margin-bottom: 15px;">
        <span style="font-size: 0.88rem; color: #475569;">
            🤖 <strong>MilkLab° RAG Assistant:</strong> ยินดีต้อนรับครับ สามารถสอบถามข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิดร้าน หรือการจัดส่งได้ทันทีครับ
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick Prompts
    st.caption("💡 คำถามพบบ่อย:")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    quick_selected = None
    with q_col1:
        if st.button("⏰ เวลาเปิด-ปิดร้าน", key="dlg_q1", use_container_width=True):
            quick_selected = "ร้านเปิดกี่โมงและปิดกี่โมง"
    with q_col2:
        if st.button("🚚 ค่าจัดส่ง & รัศมี", key="dlg_q2", use_container_width=True):
            quick_selected = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหนมีแพ็คเจลเย็นไหม"
    with q_col3:
        if st.button("🌱 เมนูสำหรับ Vegan", key="dlg_q3", use_container_width=True):
            quick_selected = "มีเจลาโต้รสไหนบ้างที่คนทานมังสวิรัติหรือวีแกนกินได้"

    prompt_to_process = quick_selected or initial_query

    # Chat Display Window
    chat_box = st.container(height=340)
    with chat_box:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 30px; color: #94A3B8; font-size: 0.9rem;">
                พิมพ์คำถามของคุณด้านล่าง หรือเลือกเมนูด้านบนเพื่อเริ่มต้นสนทนา
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
                with st.spinner("ประมวลผลคำตอบจาก Knowledge Base..."):
                    context = retrieve_top_k(final_prompt, model, index, chunks)
                    answer = generate_answer(final_prompt, context)
                st.write(answer)
                with st.expander("📄 ตรวจสอบบริบทข้อมูลอ้างอิง (RAG Context)"):
                    for i, c in enumerate(context, 1):
                        st.markdown(f"**[{i}]** `{c}`")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    # Professional Enterprise Layout Settings
    st.set_page_config(
        page_title="MilkLab° Gelato | Artisan Ice Cream",
        page_icon="🍨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Professional CSS System Design (Clean Commercial E-Commerce Theme)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'IBM Plex Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Top Navbar */
    .navbar-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        padding: 16px 32px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    
    .brand-logo {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .brand-tag {
        font-size: 0.72rem;
        background: #EFF6FF;
        color: #2563EB;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        color: #475569;
        font-weight: 500;
    }
    
    .dot-green {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
    }
    
    /* Metric KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    
    .kpi-title {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Product Card Styling */
    .product-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .product-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        transform: translateY(-2px);
    }
    
    .product-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    
    .product-icon {
        font-size: 2.5rem;
        background: #F1F5F9;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
    }
    
    .product-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    
    .product-en {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 12px;
        font-weight: 400;
    }
    
    .product-desc {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.55;
        margin-bottom: 16px;
    }
    
    .tag-pill {
        display: inline-flex;
        align-items: center;
        background: #F8FAFC;
        color: #334155;
        border: 1px solid #E2E8F0;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .price-tag {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2563EB;
    }
    
    .price-unit {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 400;
    }
    
    /* Secondary Banner */
    .banner-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 20px;
        padding: 36px 48px;
        color: #FFFFFF;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    
    .banner-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    .banner-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        max-width: 650px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load Model & FAISS Index
    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"Error loading FAISS Index: {exc}")
        st.stop()

    # Commercial Top Header Navbar
    st.markdown("""
    <div class="navbar-container">
        <div class="brand-logo">
            🍨 MilkLab° <span style="font-weight: 400; color: #64748B;">Gelato</span>
            <span class="brand-tag">Artisan 100%</span>
        </div>
        <div class="status-indicator">
            <span class="dot-green"></span>
            <span>Store Open (16:00 - 23:00) &nbsp;|&nbsp; RAG Engine Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Hero Banner Section
    st.markdown("""
    <div class="banner-hero">
        <div class="banner-title">MilkLab° Homemade Gelato Showcase</div>
        <div class="banner-subtitle">
            ไอศกรีมเจลาโต้เนื้อเนียนนุ่ม สไตล์อาร์ติซาน ใช้วัตถุดิบนำเข้าเกรดพรีเมียม 100% พร้อมระบบค้นหาข้อมูลสินค้าด้วย AI Vector Search (RAG)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Metrics Bar (KPI Cards)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Signature Flavors</div>
            <div class="kpi-value">5 Flavors</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Guaranteed Safety</div>
            <div class="kpi-value">100% Nut-Free</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Delivery Radius</div>
            <div class="kpi-value">5 km Express</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Dietary Options</div>
            <div class="kpi-value">Vegan & Sorbet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # Section Title & AI Trigger Button
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown("## 🍨 Catalog & Product Specifications")
        st.caption("คลิกเลือกปุ่ม 'สอบถามข้อมูลเมนูนี้' เพื่อเรียกใช้งาน AI Concierge ประมวลผลคำตอบอัตโนมัติ")
    with head_col2:
        if st.button("💬 เปิด AI Concierge Assistant", type="primary", use_container_width=True):
            open_ai_concierge_dialog(model, index, chunks)

    # Filter Category Tabs
    selected_tab = st.radio(
        "กรองหมวดหมู่สินค้า:",
        ["ทั้งหมด (All Flavors)", "Milk Series (สูตรใส่นม)", "Sorbet (Vegan / ปลอดนม)"],
        horizontal=True
    )

    # Filter Gelato Items
    filtered_items = GELATO_CATALOG
    if "Milk Series" in selected_tab:
        filtered_items = [x for x in GELATO_CATALOG if x["category"] == "Milk Series"]
    elif "Sorbet" in selected_tab:
        filtered_items = [x for x in GELATO_CATALOG if x["category"] == "Sorbet (Vegan)"]

    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    # Product Cards Grid (3 Columns)
    grid_cols = st.columns(3)

    for idx, item in enumerate(filtered_items):
        target_col = grid_cols[idx % 3]
        with target_col:
            # Build Tags HTML
            tags_html = ""
            for tag in item["tags"]:
                tags_html += f'<span class="tag-pill">{tag}</span>'

            card_markup = f"""
            <div class="product-card">
                <div>
                    <div class="product-header">
                        <div class="product-icon">{item["icon"]}</div>
                        <div style="text-align: right;">
                            <div class="price-tag">{item["price"]} ฿</div>
                            <div class="price-unit">/{item["size"]} ({item["calories"]})</div>
                        </div>
                    </div>
                    <div class="product-name">{item["name"]}</div>
                    <div class="product-en">{item["en_name"]}</div>
                    <div style="margin-bottom: 10px;">{tags_html}</div>
                    <div class="product-desc">{item["desc"]}</div>
                </div>
            </div>
            """
            st.markdown(card_markup, unsafe_allow_html=True)
            
            # Interactive Action Button
            if st.button(f"🔍 สอบถามข้อมูลเมนูนี้", key=f"btn_prod_{item['id']}", use_container_width=True):
                open_ai_concierge_dialog(model, index, chunks, initial_query=item["query"])
            
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Commercial Footer
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 0; color: #64748B; font-size: 0.85rem;">
        <div>© 2026 MilkLab° Gelato Co., Ltd. All Rights Reserved.</div>
        <div>Engineered with Streamlit • FAISS Vector Index • Gemini 2.5 Flash</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
