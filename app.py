"""MilkLab° Gelato - High-End Artisan Hero Landing Page Experience.
Reference Style: Berry Burst / Creamy Warm Pastel / Floating Ingredients Showcase

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
    if not api_key and hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass

    if not api_key:
        return "⚠️ กรุณาตั้งค่า GOOGLE_API_KEY ในระบบก่อนใช้งาน AI Assistant ครับ"

    client = genai.Client(api_key=api_key)
    context_str = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""คุณคือ ultrasmoothhh AI ผู้ช่วยบริการลูกค้าของแบรนด์ไอศกรีมเจลาโต้ MilkLab° Gelato (ตอบเป็นภาษาไทยอย่างสุภาพ น่ารัก กระชับ เป็นกันเอง)
โปรดตอบคำถามโดยอ้างอิงจากข้อมูลบริบท (Context) ที่กำหนดให้ต่อไปนี้เท่านั้น
หากในข้อมูลบริบทไม่มีข้อมูลที่จะตอบคำถามได้ ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลดังกล่าวในระบบ"

บริบท (Context):
{context_str}

คำถามจากลูกค้า: {query}
"""

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
    last_error = None

    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as exc:
            last_error = exc
            time.sleep(0.3)
            continue

    err_msg = str(last_error)
    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
        return "⚠️ ขณะนี้โควตาใช้งาน Gemini API ชั่วคราวเต็ม (Quota Exceeded) กรุณารอแล้วลองถามอีกครั้งนะครับ"
    elif "503" in err_msg or "UNAVAILABLE" in err_msg:
        return "⚠️ ขณะนี้เซิร์ฟเวอร์ Gemini API กำลังประมวลผลคำถามจำนวนมาก กรุณาลองถามใหม่อีกครั้งในครู่เดียวนะครับ"

    return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {last_error}"


# Gelato Catalog Data for Berry Burst Hero Style
GELATO_ITEMS = [
    {
        "id": "strawberry",
        "title1": "BERRY",
        "title2": "BURST",
        "sub_tag": "HEART SOOTHING DESSERT",
        "th_name": "เจลาโต้สตรอว์เบอร์รีซอร์เบต์",
        "price": 85,
        "size": "120g",
        "calories": "110 kcal",
        "bg_color": "#F7EFE9",          # Soft Creamy Warm Pastel
        "text_color": "#4A154B",        # Deep Berry Plum
        "accent_color": "#FF85C0",      # Vibrant Pink Glow
        "photo_url": "https://www.mealgarden.com/media/recipe/2018/01/f3uc5ts79akmznd1qi0r.jpeg", # Berry ice cream scoops
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🍓 Fresh Strawberries"],
        "desc": "สตรอว์เบอร์รีสด 100% ให้รสสัมผัสเปรี้ยวหวานฉ่ำ หอมสดชื่นละมุนหัวใจ ปลอดนม ปลอดไขมันเนย",
        "query": "ขอข้อมูลเจลาโต้สตรอว์เบอร์รีซอร์เบต์ รสชาติ สารแพ้อาหาร และราคา"
    },
    {
        "id": "hokkaido",
        "title1": "HOKKAIDO",
        "title2": "DREAM",
        "sub_tag": "ORGANIC PURE MILK DESSERT",
        "th_name": "เจลาโต้นมสดฮอกไกโด",
        "price": 80,
        "size": "120g",
        "calories": "160 kcal",
        "bg_color": "#FFF9F2",          # Milky Cream Pastel
        "text_color": "#5D4037",        # Soft Roasted Coffee
        "accent_color": "#F59E0B",      # Warm Honey Amber
        "photo_url": "https://images.unsplash.com/photo-1570197788417-0e82375c9371?q=80&w=800&auto=format&fit=crop", # Vanilla Milk scoops
        "tags": ["🥛 100% Hokkaido Milk", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "desc": "นมสดฮอกไกโดแท้ 100% นำเข้าจากญี่ปุ่น รสชาตินุ่มละมุน เข้มข้น หอมกลมกล่อมเอกลักษณ์อันดับ #1",
        "query": "ขอรายละเอียดเจลาโต้นมสดฮอกไกโด สารแพ้อาหาร และราคา"
    },
    {
        "id": "chocolate",
        "title1": "MIDNIGHT",
        "title2": "COCOA",
        "sub_tag": "INTENSE VALRHONA 70%",
        "th_name": "เจลาโต้ดาร์กช็อกโกแลต",
        "price": 85,
        "size": "120g",
        "calories": "175 kcal",
        "bg_color": "#F3EFEA",          # Warm Cocoa Dust
        "text_color": "#2C1B18",        # Midnight Cocoa
        "accent_color": "#D97706",      # Rich Amber
        "photo_url": "https://images.unsplash.com/photo-1580915411954-282cb1b0d780?q=80&w=800&auto=format&fit=crop", # Dark chocolate scoop
        "tags": ["🍫 Dark Cocoa 70%", "🥜 Nut-Free", "🌾 Gluten-Free"],
        "desc": "ดาร์กช็อกโกแลตเข้มข้น 70% ให้รสสัมผัสขมนิดๆ หวานกำลังดี เข้มข้นลึกซึ้ง ไร้สารสังเคราะห์",
        "query": "ขอรายละเอียดเจลาโต้ดาร์กช็อกโกแลต มีส่วนผสมอะไรบ้าง"
    },
    {
        "id": "matcha",
        "title1": "KYOTO",
        "title2": "MATCHA",
        "sub_tag": "CEREMONIAL UJI TEA",
        "th_name": "เจลาโต้ชาเขียวมัทฉะ",
        "price": 90,
        "size": "120g",
        "calories": "145 kcal",
        "bg_color": "#F1F8F3",          # Soft Green Tea Pastel
        "text_color": "#1B3B2B",        # Deep Forest Green
        "accent_color": "#10B981",      # Emerald Spark
        "photo_url": "https://www.allrecipes.com/thmb/totJUia-TjrmF6VnYGHOM5hVjqQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/241759-matcha-green-tea-ice-cream-VAT-003-4x3-01closeup-692d327cc2174abb84b440568f61e29a.jpg", # Matcha scoop
        "tags": ["🍵 Uji Ceremonial", "🥜 Nut-Free", "👑 Premium Grade"],
        "desc": "ผงมัทฉะเกรดพิธีการจากอุจิ เกียวโต ให้กลิ่นหอมอูมามิแท้ๆ ผสานความนุ่มเนียนของนมสดเกรดพรีเมียม",
        "query": "เจลาโต้ชาเขียวมัทฉะราคาเท่าไหร่ ใชวัตถุดิบจากไหน"
    },
    {
        "id": "mango",
        "title1": "MANGO",
        "title2": "SPLASH",
        "sub_tag": "FULL MOON PARTY TROPICAL",
        "th_name": "เจลาโต้มะม่วงมหาชนกซอร์เบต์",
        "price": 80,
        "size": "120g",
        "calories": "105 kcal",
        "bg_color": "#FEFCE8",          # Tropical Sun Pastel
        "text_color": "#78350F",        # Deep Mango Amber
        "accent_color": "#F59E0B",      # Golden Mango Glow
        "photo_url": "https://www.thedeliciouscrescent.com/wp-content/uploads/2020/10/Mango-Gelato-5.jpg", # Mango Sorbet scoop
        "tags": ["🌱 100% Vegan", "🥛 Dairy-Free", "🥭 Fresh Mango"],
        "desc": "เนื้อมะม่วงมหาชนกสดหวานฉ่ำ 100% หอมกลิ่นมะม่วงสุกแท้ๆ ปลอด dairy ทานแล้วสดชื่นกระปรี้กระเปร่า",
        "query": "ขอข้อมูลเจลาโต้มะม่วงมหาชนกซอร์เบต์ ราคาและส่วนผสม"
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
        page_title="MilkLab° Gelato | Artisan Experience",
        page_icon="🍨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    if "flavor_idx" not in st.session_state:
        st.session_state.flavor_idx = 0

    curr = GELATO_ITEMS[st.session_state.flavor_idx]

    # High-End Berry Burst & Minimal Liquid Glass Architecture (Matching Reference Image 100%)
    css_code = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700;800;900&family=Kanit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Kanit', 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Background Warm Creamy Pastel Shift */
    .stApp {{
        background-color: {curr["bg_color"]} !important;
        transition: background-color 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    /* 💎 MINIMAL LIQUID GLASS BUTTONS STYLING */
    div.stButton > button {{
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 40px !important;
        color: {curr["text_color"]} !important;
        font-family: 'Kanit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }}
    
    div.stButton > button:hover {{
        transform: translateY(-4px) scale(1.04) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border-color: #FFFFFF !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
        color: #E11D48 !important;
    }}
    
    div.stButton > button[kind="primary"] {{
        background: {curr["accent_color"]} !important;
        border: 2px solid #FFFFFF !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12) !important;
    }}
    
    div.stButton > button[kind="primary"]:hover {{
        background: {curr["text_color"]} !important;
        color: #FFFFFF !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.25) !important;
        transform: translateY(-5px) scale(1.05) !important;
    }}
    
    /* 🌟 ULTRA-MINIMALIST TOPBAR & BRAND LOGO ARCHITECTURE */
    .hanci-topbar-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        padding: 12px 0 15px 0;
    }}
    
    .hanci-topbar-pill {{
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 50px;
        padding: 10px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 95%;
        max-width: 1100px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
    }}
    
    .minimal-brand-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Fredoka', 'Kanit', sans-serif;
        user-select: none;
    }}
    
    .logo-mark {{
        width: 38px;
        height: 38px;
        background: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
    }}
    
    .logo-text {{
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: {curr["text_color"]};
        line-height: 1;
    }}
    
    .logo-degree {{
        color: #E11D48;
        font-weight: 900;
    }}
    
    .logo-sub {{
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 2.5px;
        color: {curr["text_color"]};
        margin-left: 4px;
        text-transform: uppercase;
        opacity: 0.85;
    }}
    
    .topbar-icons {{
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 1.15rem;
        color: {curr["text_color"]};
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

    /* 🍨 HERO BERRY BURST CENTERPIECE SHOWCASE */
    .hero-burst-container {{
        position: relative;
        text-align: center;
        padding: 20px 20px 50px 20px;
        min-height: 540px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    
    .giant-burst-text {{
        font-family: 'Fredoka', 'Kanit', sans-serif;
        font-size: 8.5rem;
        font-weight: 900;
        line-height: 0.82;
        letter-spacing: -1px;
        color: {curr["text_color"]};
        text-transform: uppercase;
        user-select: none;
        margin: 0;
    }}
    
    /* OVERLAPPING GELATO SCOOP BOWL IN CENTER */
    .center-gelato-bowl-frame {{
        position: relative;
        width: 320px;
        height: 320px;
        margin: -95px auto -75px auto;
        z-index: 5;
    }}
    
    .gelato-bowl-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
        box-shadow: 0 25px 60px rgba(0,0,0,0.18);
        border: 6px solid #FFFFFF;
        animation: floatBowl 5s ease-in-out infinite;
    }}
    
    /* FLOATING INGREDIENT ELEMENTS */
    .floating-leaf-1 {{
        position: absolute;
        left: 14%;
        top: 25%;
        font-size: 2.8rem;
        animation: floatLeaf 4s ease-in-out infinite;
        opacity: 0.9;
    }}
    
    .floating-leaf-2 {{
        position: absolute;
        right: 15%;
        top: 18%;
        font-size: 2.5rem;
        animation: floatLeaf 5s ease-in-out infinite reverse;
        opacity: 0.9;
    }}
    
    .floating-berry-1 {{
        position: absolute;
        left: 20%;
        bottom: 12%;
        font-size: 3rem;
        animation: floatLeaf 4.5s ease-in-out infinite;
    }}
    
    .sub-tagline-burst {{
        font-family: 'Kanit', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: {curr["text_color"]};
        text-transform: uppercase;
        margin-top: 15px;
        margin-bottom: 22px;
        opacity: 0.9;
    }}

    @keyframes floatBowl {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-12px) rotate(2deg); }}
    }}
    
    @keyframes floatLeaf {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-15px) rotate(10deg); }}
    }}

    /* 📌 FIXED FLOATING CIRCULAR CHATBOT POPUP (BOTTOM-LEFT) */
    div[data-testid="stPopover"], .stPopover {{
        position: fixed !important;
        bottom: 30px !important;
        left: 30px !important;
        right: auto !important;
        width: 75px !important;
        height: 75px !important;
        z-index: 999999 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stPopover"] button, .stPopover button, button[data-testid="stBaseButton-popover"] {{
        width: 75px !important;
        min-width: 75px !important;
        max-width: 75px !important;
        height: 75px !important;
        min-height: 75px !important;
        max-height: 75px !important;
        border-radius: 50% !important;
        background: #FFFFFF !important;
        border: 3.5px solid {curr["text_color"]} !important;
        box-shadow: 0 14px 40px rgba(0,0,0,0.28) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        color: {curr["text_color"]} !important;
        padding: 0 !important;
        margin: 0 !important;
        animation: pulseFloating 3.5s infinite ease-in-out !important;
    }}
    
    div[data-testid="stPopover"] button:hover, .stPopover button:hover {{
        transform: scale(1.18) rotate(10deg) !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.4) !important;
        background: #FFFFFF !important;
    }}

    div[data-testid="stPopover"] button p {{
        font-size: 2.3rem !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }}

    /* Hide dropdown caret icon next to popover button */
    div[data-testid="stPopover"] button svg,
    div[data-testid="stPopover"] button [data-testid="stIcon"],
    div[data-testid="stPopover"] button [data-testid="stBaseButton-popoverIcon"] {{
        display: none !important;
    }}

    div[data-testid="stPopoverBody"] {{
        border-radius: 26px !important;
        padding: 24px !important;
        border: 2px solid rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 24px 70px rgba(0,0,0,0.25) !important;
        background: rgba(255, 255, 255, 0.96) !important;
        backdrop-filter: blur(24px) !important;
        width: 480px !important;
        max-width: 92vw !important;
    }}

    /* 📜 INFINITE SCROLLING MARQUEE TICKER ANIMATION */
    .marquee-wrapper {{
        width: 100%;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 14px 0;
        margin: 25px 0 15px 0;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95);
        border-bottom: 1.5px solid rgba(255, 255, 255, 0.95);
        box-shadow: 0 6px 25px rgba(0,0,0,0.03);
    }}
    
    .marquee-track {{
        display: flex;
        align-items: center;
        gap: 35px;
        white-space: nowrap;
        font-family: 'Fredoka', 'Kanit', sans-serif;
        font-weight: 800;
        font-size: 0.98rem;
        letter-spacing: 2px;
        color: {curr["text_color"]};
        text-transform: uppercase;
        animation: infiniteScrollMarquee 30s linear infinite;
    }}

    /* 🌟 SCROLL REVEAL MOTION ANIMATIONS */
    html {{
        scroll-behavior: smooth !important;
    }}

    .scroll-reveal-card {{
        animation: cardRevealFade 0.85s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.4s ease !important;
    }}

    .scroll-reveal-card:hover {{
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 0 20px 45px rgba(0,0,0,0.12) !important;
    }}

    @keyframes cardRevealFade {{
        0% {{
            opacity: 0;
            transform: translateY(35px) scale(0.96);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0px) scale(1);
        }}
    }}

    @keyframes infiniteScrollMarquee {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}

    @keyframes pulseFloating {{
        0%, 100% {{ transform: translateY(0px) scale(1); }}
        50% {{ transform: translateY(-8px) scale(1.05); }}
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

    # 1. ULTRA-MINIMALIST TOP BAR
    hanci_topbar_html = """<div class="hanci-topbar-wrapper">
<div class="hanci-topbar-pill">
<div class="minimal-brand-logo">
<div class="logo-mark">🍨</div>
<div class="logo-text">MILKLAB<span class="logo-degree">°</span> <span class="logo-sub">GELATO</span></div>
</div>
<div class="topbar-icons">
<span class="topbar-icon" title="Search">🔍</span>
<span class="topbar-icon" title="Wishlist">♡</span>
<span class="topbar-icon-cart" title="Cart">🛍️ <span class="cart-count-badge">0</span></span>
</div>
</div>
</div>"""
    st.markdown(hanci_topbar_html, unsafe_allow_html=True)

    # 2. FLAVOR SELECTOR PILL BUTTONS
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    sel_cols = st.columns(len(GELATO_ITEMS))

    for idx, item in enumerate(GELATO_ITEMS):
        with sel_cols[idx]:
            is_active = (idx == st.session_state.flavor_idx)
            label = f"✨ {item['title1']} {item['title2']}" if is_active else f"{item['title1']} {item['title2']}"
            if st.button(label, key=f"btn_flv_{item['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.flavor_idx = idx
                st.rerun()

    # 3. HERO BERRY BURST CENTERPIECE SHOWCASE
    hero_html = f"""<div class="hero-burst-container">
<div class="floating-leaf-1">🍃</div>
<div class="floating-leaf-2">🌿</div>
<div class="floating-berry-1">🫐</div>
<div class="giant-burst-text">{curr["title1"]}</div>
<div class="center-gelato-bowl-frame">
<img src="{curr['photo_url']}" class="gelato-bowl-img" alt="Gelato Scoop Bowl" />
</div>
<div class="giant-burst-text">{curr["title2"]}</div>
<div class="sub-tagline-burst">{curr["sub_tag"]} &nbsp;•&nbsp; PRICE: {curr["price"]} THB ({curr["size"]})</div>
</div>"""
    st.markdown(hero_html, unsafe_allow_html=True)

    # Action Button under Hero
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(f"🛍️ ORDER NOW ({curr['th_name']})", key="btn_order_now", type="primary", use_container_width=True):
            st.toast(f"🛒 เพิ่ม {curr['th_name']} ลงในรายการสั่งซื้อเรียบร้อยแล้ว!")

    # 📜 INFINITE SCROLLING MARQUEE TICKER BANNER
    marquee_html = f"""<div class="marquee-wrapper">
<div class="marquee-track">
<span>🍨 MILKLAB° GELATO</span><span>•</span>
<span>🥛 100% HOKKAIDO MILK</span><span>•</span>
<span>🍫 VALRHONA 70% DARK COCOA</span><span>•</span>
<span>🍵 CEREMONIAL UJI MATCHA</span><span>•</span>
<span>🍓 FRESH STRAWBERRY SORBET</span><span>•</span>
<span>🥭 100% VEGAN MANGO SORBET</span><span>•</span>
<span>🧊 COLD-PACK GEL DELIVERY 45 MINS</span><span>•</span>
<span>🍨 MILKLAB° GELATO</span><span>•</span>
<span>🥛 100% HOKKAIDO MILK</span><span>•</span>
<span>🍫 VALRHONA 70% DARK COCOA</span><span>•</span>
<span>🍵 CEREMONIAL UJI MATCHA</span><span>•</span>
<span>🍓 FRESH STRAWBERRY SORBET</span><span>•</span>
<span>🥭 100% VEGAN MANGO SORBET</span><span>•</span>
<span>🧊 COLD-PACK GEL DELIVERY 45 MINS</span><span>•</span>
</div>
</div>"""
    st.markdown(marquee_html, unsafe_allow_html=True)

    # 4. ⭐ NEW SECTION 1: SOCIAL PROOF & RATING BANNER
    social_proof_html = f"""<div class="scroll-reveal-card" style="background: rgba(255,255,255,0.75); backdrop-filter: blur(16px); border-radius: 20px; padding: 18px 24px; text-align: center; max-width: 850px; margin: 25px auto 10px auto; border: 1.5px solid rgba(255,255,255,0.95); box-shadow: 0 10px 30px rgba(0,0,0,0.04);">
<div style="font-size: 1.15rem; font-weight: 900; color: #F59E0B; letter-spacing: 1.5px;">★★★★★ 4.9 / 5.0 RATING</div>
<div style="font-weight: 700; color: {curr['text_color']}; font-size: 0.98rem; margin: 5px 0 3px 0;">"เนื้อสัมผัสเจลาโต้เนียนนุ่ม รสชาติเข้มข้นสดชื่นที่สุดเท่าที่เคยทานมา! อร่อยประทับใจมาก"</div>
<div style="font-size: 0.78rem; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 1.5px;">GARNERED FROM 1,200+ GELATO LOVERS IN THAILAND</div>
</div>"""
    st.markdown(social_proof_html, unsafe_allow_html=True)

    # 5. 🚚 NEW SECTION 2: INSTANT DELIVERY & COLD-PACK GUARANTEE CARD
    delivery_html = f"""<div class="scroll-reveal-card" style="background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,255,255,0.65)); backdrop-filter: blur(20px); border-radius: 24px; padding: 28px; border: 2px solid #FFFFFF; box-shadow: 0 15px 35px rgba(0,0,0,0.06); margin: 35px 0;">
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px; text-align: center;">
<div style="flex: 1; min-width: 200px;">
<div style="font-size: 2.2rem; margin-bottom: 6px;">🧊</div>
<div style="font-weight: 800; font-size: 1.05rem; color: #0F172A;">Cold-Pack Gel Guarantee</div>
<div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">แพ็คด้วยเจลเย็นคงความเย็น 45 นาที ไม่ละลายแน่นอน 100%</div>
</div>
<div style="flex: 1; min-width: 200px; border-left: 1px solid rgba(0,0,0,0.08); border-right: 1px solid rgba(0,0,0,0.08); padding: 0 15px;">
<div style="font-size: 2.2rem; margin-bottom: 6px;">🚚</div>
<div style="font-weight: 800; font-size: 1.05rem; color: #0F172A;">Speedy 5 km Delivery</div>
<div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">จัดส่งด่วนรัศมี 5 กม. ค่าจัดส่งเพียง 30 บาททั่วเขต</div>
</div>
<div style="flex: 1; min-width: 200px;">
<div style="font-size: 2.2rem; margin-bottom: 6px;">🛍️</div>
<div style="font-weight: 800; font-size: 1.05rem; color: #0F172A;">No Minimum Order</div>
<div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">ไม่มีขั้นต่ำในการสั่งซื้อ สั่ง 1 ถ้วยก็พร้อมจัดส่งทันที</div>
</div>
</div>
</div>"""
    st.markdown(delivery_html, unsafe_allow_html=True)

    st.markdown("---")

    # 6. PRODUCT CARDS GRID
    st.markdown("## 🎴 ALL ARTISAN GELATO FLAVORS (เลือกชมเจลาโต้ทุกรสชาติ)")
    grid_cols = st.columns(3)

    for idx, item in enumerate(GELATO_ITEMS):
        target_col = grid_cols[idx % 3]
        with target_col:
            tags_html = "".join(f'<span style="background:rgba(255,255,255,0.7); color:{item["text_color"]}; padding:4px 10px; border-radius:10px; font-size:0.78rem; font-weight:700; margin-right:4px; margin-bottom:4px; display:inline-block;">{t}</span>' for t in item["tags"])
            
            card_html = f"""<div class="scroll-reveal-card" style="background: {item['bg_color']}; border-radius: 24px; padding: 24px; text-align: center; box-shadow: 0 12px 30px rgba(0,0,0,0.06); margin-bottom: 15px; border: 2px solid rgba(255,255,255,0.85);">
<div style="height: 140px; border-radius: 16px; overflow: hidden; margin-bottom: 14px;">
<img src="{item['photo_url']}" style="width: 100%; height: 100%; object-fit: cover;" />
</div>
<div style="font-family: 'Fredoka', sans-serif; font-size: 1.9rem; color: {item['text_color']}; line-height: 1; font-weight: 800;">{item['title1']} {item['title2']}</div>
<div style="font-weight: 700; color: {item['text_color']}; font-size: 0.95rem; margin-top: 4px; margin-bottom: 8px;">{item['th_name']}</div>
<div style="font-size: 0.82rem; color: #475569; line-height: 1.4; margin-bottom: 12px; height: 38px; overflow: hidden;">{item['desc']}</div>
<div style="margin-bottom: 12px;">{tags_html}</div>
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.6); padding: 8px 14px; border-radius: 14px; margin-bottom: 12px;">
<span style="font-size: 0.8rem; font-weight: 700; color: #64748B;">⚡ {item['calories']}</span>
<span style="font-size: 1.25rem; font-weight: 900; color: {item['text_color']};">{item['price']} THB <span style="font-size: 0.75rem; font-weight: 500;">/ {item['size']}</span></span>
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button(f"✨ เลือกรสชาตินี้ ({item['th_name']})", key=f"btn_grid_sel_{item['id']}", use_container_width=True):
                st.session_state.flavor_idx = idx
                st.rerun()

            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 📌 EXCLUSIVE FLOATING CIRCULAR CHATBOT POPUP (BOTTOM-LEFT): ULTRASMOOTHHH AI
    with st.popover("🍨", help="คลิกเพื่อถามน้อง ultrasmoothhh ได้เลยน้าา"):
        st.markdown("### 🍨 ถามน้อง ultrasmoothhh ได้เลยน้าา")
        st.caption("ผู้ช่วย AI ประจำร้าน MilkLab° Gelato สอบถามเมนู เวลาเปิด-ปิด และการจัดส่งได้เลยครับ")
        
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

        pop_chat_box = st.container(height=360)
        with pop_chat_box:
            if not st.session_state.messages:
                st.info("👋 สวัสดีครับ! มีอะไรสงสัยเกี่ยวกับเจลาโต้ ถามน้อง ultrasmoothhh ได้เลยน้าา 😊")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        pop_input = st.chat_input("ถามน้อง ultrasmoothhh ได้เลยน้าา...")
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

    # 8. 🖤 NEW SECTION 4: HIGH-FASHION MINIMALIST FOOTER
    footer_html = """<footer style="background: #0F172A; border-radius: 30px 30px 0 0; padding: 45px 35px 30px 35px; color: #F8FAFC; margin-top: 50px;">
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 30px; margin-bottom: 30px;">
<div>
<div style="font-family: 'Fredoka', sans-serif; font-size: 2rem; font-weight: 800; color: #FFFFFF;">MILKLAB<span style="color:#E11D48;">°</span> GELATO</div>
<div style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; max-width: 320px; line-height: 1.5;">
ไอศกรีมเจลาโต้โฮมเมดพรีเมียม วัตถุดิบธรรมชาติสดใหม่ 100% สดชื่นละมุนหัวใจในทุกๆ คำ
</div>
</div>
<div>
<div style="font-weight: 800; font-size: 0.95rem; color: #F8FAFC; text-transform: uppercase; letter-spacing: 1px;">⏰ เวลาเปิด-ปิดร้าน</div>
<div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 8px;">เปิดบริการทุกวัน (ยกเว้นวันจันทร์)</div>
<div style="font-size: 0.88rem; color: #E11D48; font-weight: 700; margin-top: 4px;">16:00 - 23:00 น.</div>
</div>
<div>
<div style="font-weight: 800; font-size: 0.95rem; color: #F8FAFC; text-transform: uppercase; letter-spacing: 1px;">📍 พิกัด & ติดต่อ</div>
<div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 8px;">ตั้งอยู่หน้ามหาวิทยาลัย</div>
<div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 4px;">สั่งซื้อล่วงหน้าผ่าน LINE OA (ก่อน 15:00 น.)</div>
</div>
</div>
<div style="border-top: 1px solid #334155; padding-top: 20px; text-align: center; font-size: 0.8rem; color: #64748B;">
© 2026 MILKLAB° GELATO. ALL RIGHTS RESERVED. POWERED BY GEMINI RAG AI 🍨
</div>
</footer>"""
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
