"""MilkLab° Gelato - High-Fashion DTC Animated Experience.
Reference Style: Boba Ice Cream / Full Moon Party / Karaoke Night

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


# Gelato Catalog with Dynamic Themes & High-Fashion DTC Aesthetics
GELATO_FLAVORS = [
    {
        "id": "strawberry",
        "name": "KARAOKE NIGHT",
        "sub_name": "STRAWBERRY SHORTCAKE SORBET",
        "th_name": "เจลาโต้สตรอว์เบอร์รีซอร์เบต์",
        "price": 85,
        "size": "120g",
        "bg_color": "#FFC5D3",
        "card_bg": "#FF8DA1",
        "accent_color": "#D81B60",
        "text_color": "#880E4F",
        "tag_bg": "rgba(216, 27, 96, 0.15)",
        "icon": "🍓",
        "calories": "110 kcal",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥜 Nut-Free"],
        "desc": "สตรอว์เบอร์รีสด 100% ให้รสสัมผัสเปรี้ยวหวานฉ่ำ เติมความสดชื่นยามค่ำคืนสไตล์ Karaoke Night ปลอดนม ปลอดไขมันเนย",
        "query": "ขอข้อมูลเจลาโต้สตรอว์เบอร์รีซอร์เบต์ รสชาติ สารแพ้อาหาร และราคา"
    },
    {
        "id": "hokkaido",
        "name": "HOKKAIDO DREAM",
        "sub_name": "PURE HOKKAIDO FRESH MILK",
        "th_name": "เจลาโต้นมสดฮอกไกโด",
        "price": 80,
        "size": "120g",
        "bg_color": "#FFEAD2",
        "card_bg": "#F5C7A9",
        "accent_color": "#D97706",
        "text_color": "#78350F",
        "tag_bg": "rgba(217, 119, 6, 0.15)",
        "icon": "🍦",
        "calories": "195 kcal",
        "tags": ["🥛 100% Hokkaido Milk", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "desc": "นมสดฮอกไกโดแท้ 100% นำเข้าจากญี่ปุ่น รสชาตินุ่มละมุน เข้มข้น หอมกลมกล่อมเอกลักษณ์อันดับ #1 ของ MilkLab°",
        "query": "ขอรายละเอียดเจลาโต้นมสดฮอกไกโด สารแพ้อาหาร และราคา"
    },
    {
        "id": "chocolate",
        "name": "MIDNIGHT COCOA",
        "sub_name": "DARK CHOCOLATE 70%",
        "th_name": "เจลาโต้ดาร์กช็อกโกแลต",
        "price": 85,
        "size": "120g",
        "bg_color": "#D8C4B6",
        "card_bg": "#4F3B32",
        "accent_color": "#B45309",
        "text_color": "#3E2723",
        "tag_bg": "rgba(79, 59, 50, 0.2)",
        "icon": "🍫",
        "calories": "210 kcal",
        "tags": ["🍫 Dark Cocoa 70%", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "desc": "ดาร์กช็อกโกแลตเข้มข้น 70% ให้รสสัมผัสขมนิดๆ หวานกำลังดี เข้มข้นลึกซึ้ง ไร้สารสังเคราะห์",
        "query": "ขอรายละเอียดเจลาโต้ดาร์กช็อกโกแลต มีส่วนผสมอะไรบ้าง"
    },
    {
        "id": "matcha",
        "name": "KYOTO CEREMONY",
        "sub_name": "UJI MATCHA GREEN TEA",
        "th_name": "เจลาโต้ชาเขียวมัทฉะ",
        "price": 90,
        "size": "120g",
        "bg_color": "#D1E7DD",
        "card_bg": "#4D7C0F",
        "accent_color": "#3F6212",
        "text_color": "#1E3A8A",
        "tag_bg": "rgba(63, 98, 18, 0.15)",
        "icon": "🍵",
        "calories": "180 kcal",
        "tags": ["🍵 Uji Ceremonial Grade", "🥜 Nut-Free", "👑 Premium"],
        "desc": "ผงมัทฉะเกรดพิธีการจากเมืองอุจิ เกียวโต ให้ความหอมเข้มข้น มิติรสชาติลึกซึ้งแท้สไตล์ญี่ปุ่น",
        "query": "เจลาโต้ชาเขียวมัทฉะราคาเท่าไหร่ ใช้วัตถุดิบจากไหน"
    },
    {
        "id": "mango",
        "name": "FULL MOON PARTY",
        "sub_name": "MAHACHANOK MANGO SORBET",
        "th_name": "เจลาโต้มะม่วงมหาชนกซอร์เบต์",
        "price": 80,
        "size": "120g",
        "bg_color": "#FFE97D",
        "card_bg": "#F59E0B",
        "accent_color": "#B45309",
        "text_color": "#78350F",
        "tag_bg": "rgba(180, 83, 9, 0.15)",
        "icon": "🥭",
        "calories": "125 kcal",
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥭 Fresh Mango"],
        "desc": "เนื้อมะม่วงมหาชนกสดหวานฉ่ำ 100% ให้ความสดชื่นระเบิดรสชาติหวานอมเปรี้ยวคลายร้อนสไตล์ Full Moon Party",
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
        if st.button("⏰ เวลาเปิด-ปิดร้าน", key="dlg_f1", use_container_width=True):
            quick_selected = "ร้านเปิดกี่โมงและปิดกี่โมง"
    with q_col2:
        if st.button("🚚 ค่าจัดส่ง & รัศมี", key="dlg_f2", use_container_width=True):
            quick_selected = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหนมีแพ็คเจลเย็นไหม"
    with q_col3:
        if st.button("🌱 เมนูสำหรับ Vegan", key="dlg_f3", use_container_width=True):
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

    # Initialize Selected Flavor Session State
    if "selected_flavor_idx" not in st.session_state:
        st.session_state.selected_flavor_idx = 0

    current_flavor = GELATO_FLAVORS[st.session_state.selected_flavor_idx]

    # Dynamic High-Fashion CSS with Theme Colors & Keyframe Animations
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Kanit:wght@400;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Kanit', 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Dynamic Animated Background Transition */
    .stApp {{
        background-color: {current_flavor["bg_color"]} !important;
        transition: background-color 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    /* Scroll Behavior */
    html {{
        scroll-behavior: smooth;
    }}
    
    /* High Impact Banner Header */
    .hero-banner-container {{
        text-align: center;
        padding: 40px 20px 20px 20px;
        position: relative;
    }}
    
    .hero-big-title {{
        font-family: 'Anton', 'Kanit', sans-serif;
        font-size: 5.5rem;
        font-weight: 900;
        letter-spacing: 2px;
        color: #FFFFFF;
        text-shadow: 0 10px 30px rgba(0,0,0,0.15);
        line-height: 0.95;
        text-transform: uppercase;
        animation: fadeInDown 0.8s ease-out;
    }}
    
    .hero-sub-text {{
        font-family: 'Kanit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: {current_flavor["accent_color"]};
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 15px;
    }}
    
    /* Floating Animated Ice Cream Tub Container */
    .tub-display-frame {{
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.8);
        border-radius: 36px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.12);
        animation: floatTub 4s ease-in-out infinite;
        margin: 20px 0;
    }}
    
    .tub-icon-large {{
        font-size: 7.5rem;
        filter: drop-shadow(0 15px 25px rgba(0,0,0,0.15));
        transition: transform 0.5s ease;
    }}
    
    .tub-icon-large:hover {{
        transform: scale(1.15) rotate(5deg);
    }}
    
    .flavor-title-display {{
        font-family: 'Anton', 'Kanit', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        color: {current_flavor["accent_color"]};
        line-height: 1.1;
        letter-spacing: 1px;
        margin-top: 15px;
    }}
    
    .flavor-desc-display {{
        font-size: 1.1rem;
        color: #334155;
        max-width: 650px;
        margin: 15px auto;
        line-height: 1.6;
        font-weight: 500;
    }}
    
    .price-pill-large {{
        display: inline-block;
        background: #FFFFFF;
        color: {current_flavor["accent_color"]};
        font-size: 1.8rem;
        font-weight: 800;
        padding: 10px 28px;
        border-radius: 30px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border: 2px solid {current_flavor["accent_color"]};
    }}
    
    /* High Fashion Flavor Showcase Cards Grid */
    .dtc-card-frame {{
        background: #FFFFFF;
        border-radius: 28px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.06);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: 3px solid transparent;
        height: 100%;
    }}
    
    .dtc-card-frame:hover {{
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 45px rgba(0,0,0,0.12);
        border-color: {current_flavor["accent_color"]};
    }}
    
    .dtc-card-title {{
        font-family: 'Anton', 'Kanit', sans-serif;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 1px;
        color: #0F172A;
        margin-top: 10px;
        text-transform: uppercase;
    }}

    /* Keyframes for Floating Animation & Scroll */
    @keyframes floatTub {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-14px) rotate(2deg); }}
    }}
    
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
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
        border: 3px solid {current_flavor["accent_color"]} !important;
        box-shadow: 0 12px 36px rgba(0,0,0,0.2) !important;
        font-size: 2.2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        color: {current_flavor["accent_color"]} !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stPopover"] > button:hover {{
        transform: scale(1.15) rotate(10deg) !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.3) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Load FAISS Index
    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"Error loading FAISS Index: {exc}")
        st.stop()

    # 1. High Impact Hero Banner
    st.markdown(f"""
    <div class="hero-banner-container">
        <div class="hero-big-title">MILKLAB × GELATO</div>
        <div class="hero-sub-text">A MAGICAL DUO OF ARTISAN HOMEMADE ICE CREAM</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Interactive Flavor Selector Buttons (Color Theme Changing!)
    st.markdown("### 🎨 เลือกเปลี่ยนสีธีมและรสชาติไอศกรีม (Interactive Flavor Changer):")
    sel_cols = st.columns(len(GELATO_FLAVORS))

    for idx, flavor in enumerate(GELATO_FLAVORS):
        with sel_cols[idx]:
            is_active = (idx == st.session_state.selected_flavor_idx)
            btn_label = f"{flavor['icon']} {flavor['name']}"
            if is_active:
                btn_label = f"✨ {flavor['icon']} {flavor['name']}"
                
            if st.button(btn_label, key=f"flavor_sel_{flavor['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.selected_flavor_idx = idx
                st.rerun()

    # 3. Dynamic Animated Tub Spotlight Display
    st.markdown(f"""
    <div class="tub-display-frame">
        <div class="tub-icon-large">{current_flavor["icon"]}</div>
        <div class="flavor-title-display">{current_flavor["name"]}</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #475569; letter-spacing: 2px;">{current_flavor["sub_name"]}</div>
        <div class="flavor-desc-display">{current_flavor["desc"]}</div>
        <div class="price-pill-large">{current_flavor["price"]} THB <span style="font-size: 1.1rem; font-weight: 500;">/ {current_flavor["size"]} ({current_flavor["calories"]})</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Action Buttons under Spotlight
    act_col1, act_col2, act_col3 = st.columns([1, 2, 1])
    with act_col2:
        if st.button(f"✨ ถามน้อง AI เกี่ยวกับ {current_flavor['th_name']}", key="btn_hero_ask", type="primary", use_container_width=True):
            open_ai_dialog(model, index, chunks, initial_query=current_flavor["query"])

    st.markdown("---")
    st.markdown("## 🍨 EXPLORE ALL FLAVORS (ทุกรสชาติมี Animation เมื่อ Scroll & Hover)")

    # 4. Showcase Cards Grid (Reference Image 2 Style!)
    card_cols = st.columns(3)

    for idx, item in enumerate(GELATO_FLAVORS):
        target_col = card_cols[idx % 3]
        with target_col:
            tags_html = "".join(f'<span style="background:{item["tag_bg"]}; color:{item["accent_color"]}; padding:4px 10px; border-radius:10px; font-size:0.78rem; font-weight:600; margin-right:4px;">{t}</span>' for t in item["tags"])
            
            card_html = f"""
            <div class="dtc-card-frame" style="background: {item['bg_color']};">
                <div style="font-size: 4rem; margin-bottom: 10px;">{item['icon']}</div>
                <div class="dtc-card-title">{item['name']}</div>
                <div style="font-weight: 700; color: {item['accent_color']}; font-size: 0.95rem; margin-bottom: 10px;">{item['sub_name']}</div>
                <div style="margin-bottom: 12px;">{tags_html}</div>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.5; margin-bottom: 16px;">{item['desc']}</div>
                <div style="font-size: 1.5rem; font-weight: 900; color: {item['accent_color']};">{item['price']} ฿</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button(f"✨ ถาม AI ({item['th_name']})", key=f"btn_dtc_ask_{item['id']}", use_container_width=True):
                st.session_state.selected_flavor_idx = idx
                open_ai_dialog(model, index, chunks, initial_query=item["query"])

            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

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
            if st.button("⏰ เวลาเปิด-ปิดร้าน", key="p_dtc1", use_container_width=True):
                quick_pop_q = "ร้านเปิดกี่โมงและปิดกี่โมง"
        with p_c2:
            if st.button("🚚 ค่าส่ง & รัศมี", key="p_dtc2", use_container_width=True):
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

    # High Fashion Footer
    st.markdown(f"""
    <div style="text-align: center; color: #475569; font-size: 0.9rem; padding: 20px 0; font-weight: 600;">
        MILKLAB × GELATO • HIGH-FASHION DTC EXPERIENCE • POWERED BY GEMINI RAG 🍨
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
