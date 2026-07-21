"""MilkLab Gelato Dashboard & RAG Chatbot Popup (S3).

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
        return "⚠️ กรุณาตั้งค่า GOOGLE_API_KEY ก่อนใช้งาน AI Chatbot ครับ"

    client = genai.Client(api_key=api_key)
    context_str = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""คุณเป็น AI ผู้ช่วยบริการลูกค้าของร้านไอศกรีมเจลาโต้ MilkLab° Gelato (ตอบเป็นภาษาไทยอย่างสุภาพ น่ารัก กระชับ เป็นกันเอง)
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


# Gelato Menu Data
GELATO_MENUS = [
    {
        "id": "hokkaido",
        "name": "เจลาโต้นมสดฮอกไกโด",
        "en_name": "Hokkaido Milk Gelato",
        "icon": "🍦",
        "price": 80,
        "size": "120g",
        "desc": "นมสดฮอกไกโดแท้ 100% รสชาติหอมนุ่ม เข้มข้น กลมกล่อม",
        "badges": ["🥛 นมสดแท้", "🥜 Nut-Free", "✨ Signature"],
        "query": "ขอรายละเอียดเจลาโต้นมสดฮอกไกโด และสารแพ้อาหารหน่อยครับ"
    },
    {
        "id": "chocolate",
        "name": "เจลาโต้ดาร์กช็อกโกแลต",
        "en_name": "Dark Chocolate Gelato",
        "icon": "🍫",
        "price": 85,
        "size": "120g",
        "desc": "ผงโกโก้พรีเมียมเข้มข้น 70% รสชาติเข้มข้น กลมกล่อม หวานกำลังดี",
        "badges": ["🍫 โกโก้ 70%", "🥜 Nut-Free", "🔥 Bestseller"],
        "query": "ขอรายละเอียดเจลาโต้ดาร์กช็อกโกแลต มีส่วนผสมอะไรบ้าง"
    },
    {
        "id": "strawberry",
        "name": "เจลาโต้สตรอว์เบอร์รีซอร์เบต์",
        "en_name": "Strawberry Sorbet Gelato",
        "icon": "🍓",
        "price": 85,
        "size": "120g",
        "desc": "สตรอว์เบอร์รีสดแท้ 100% รสชาติหวานอมเปรี้ยว สดชื่น ปลอดนม",
        "badges": ["🌱 Vegan 100%", "🥛 ปลอดนม", "🥜 Nut-Free"],
        "query": "คนทานมังสวิรัติกินเจลาโต้สตรอว์เบอร์รีซอร์เบต์ได้ไหม"
    },
    {
        "id": "matcha",
        "name": "เจลาโต้ชาเขียวมัทฉะ",
        "en_name": "Matcha Green Tea Gelato",
        "icon": "🍵",
        "price": 90,
        "size": "120g",
        "desc": "ผงมัทฉะเกรดพิธีการนำเข้าจากอุจิ เกียวโต รสชาติเข้มข้นละมุนลิ้น",
        "badges": ["🍵 มัทฉะเกียวโต", "🥜 Nut-Free", "👑 Premium"],
        "query": "เจลาโต้ชาเขียวมัทฉะราคาเท่าไหร่ และใช้วัตถุดิบจากไหน"
    },
    {
        "id": "mango",
        "name": "เจลาโต้มะม่วงมหาชนกซอร์เบต์",
        "en_name": "Mahachanok Mango Sorbet",
        "icon": "🥭",
        "price": 80,
        "size": "120g",
        "desc": "เนื้อมะม่วงมหาชนกสดแท้ รสหวานฉ่ำ หอมกลิ่นมะม่วงธรรมชาติ ปลอดนม",
        "badges": ["🌱 Vegan 100%", "🥭 มะม่วงสด", "🥜 Nut-Free"],
        "query": "ขอข้อมูลเจลาโต้มะม่วงมหาชนกซอร์เบต์ ราคาเท่าไหร่และปลอดนมไหม"
    }
]


# Dialog Pop-up for Chatbot
@st.dialog("💬 MilkLab° AI Assistant (เจลาโต้โฮมเมด)", width="large")
def chatbot_popup_dialog(model, index, chunks, initial_query: str = ""):
    st.caption("🍨 ถามตอบข้อมูลเมนูเจลาโต้ สารแพ้อาหาร เวลาเปิด-ปิด และบริการจัดส่ง")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick Suggestion Pills
    st.markdown("##### 💡 คำถามยอดฮิตที่พบบ่อย:")
    col_q1, col_q2, col_q3 = st.columns(3)
    
    selected_quick_q = None
    with col_q1:
        if st.button("⏰ เวลาเปิด-ปิดร้าน", key="btn_q1", use_container_width=True):
            selected_quick_q = "ร้านเปิดกี่โมงและปิดกี่โมง"
    with col_q2:
        if st.button("🚚 ค่าส่งและรัศมีส่ง", key="btn_q2", use_container_width=True):
            selected_quick_q = "ค่าจัดส่งเท่าไหร่และส่งไกลแค่ไหน"
    with col_q3:
        if st.button("🌱 เมนูสำหรับ Vegan", key="btn_q3", use_container_width=True):
            selected_quick_q = "มีเมนูเจลาโต้รสไหนบ้างที่คนทานมังสวิรัติหรือวีแกนกินได้"

    # Handle auto-input from menu click or quick pill
    active_prompt = selected_quick_q or initial_query

    # Display Chat History
    st.markdown("---")
    chat_container = st.container(height=350)
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 สวัสดีครับ! น้อง AI ยินดีให้บริการ สอบถามข้อมูล MilkLab° Gelato ได้เลยครับ 😊")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # User Input
    user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับไอศกรีมเจลาโต้ที่นี่...")
    prompt_to_run = user_input or active_prompt

    if prompt_to_run:
        # Prevent double submit if same prompt
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt_to_run)

            with st.chat_message("assistant"):
                with st.spinner("กำลังค้นข้อมูลในเมนู..."):
                    context = retrieve_top_k(prompt_to_run, model, index, chunks)
                    answer = generate_answer(prompt_to_run, context)
                st.write(answer)
                with st.expander("🔍 ข้อมูลอ้างอิงจากระบบ (Source Chunks)"):
                    for i, c in enumerate(context, 1):
                        st.markdown(f"**[{i}]** {c}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    # Streamlit Page Config & Minimal Theme
    st.set_page_config(
        page_title="MilkLab° Gelato Dashboard",
        page_icon="🍨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom Minimal Pastel Styling (CSS)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    .main {
        background-color: #FFFDF9;
    }
    
    /* Cute Header Banner */
    .header-box {
        background: linear-gradient(135deg, #FFEFF2 0%, #FFF5E4 100%);
        padding: 30px 40px;
        border-radius: 24px;
        box-shadow: 0 10px 25px rgba(255, 142, 158, 0.08);
        border: 1px solid #FFE4E8;
        margin-bottom: 25px;
        text-align: center;
    }
    
    .header-title {
        color: #D84062;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .header-subtitle {
        color: #6C5B7B;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* Minimal Card */
    .menu-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 22px;
        border: 1px solid #F3E8EE;
        box-shadow: 0 8px 20px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 20px;
    }
    
    .menu-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(216, 64, 98, 0.08);
        border-color: #FFCBD5;
    }
    
    .menu-icon {
        font-size: 2.8rem;
        margin-bottom: 10px;
    }
    
    .menu-name {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2D2424;
        margin-bottom: 2px;
    }
    
    .menu-en {
        font-size: 0.85rem;
        color: #9A8C98;
        margin-bottom: 12px;
    }
    
    .menu-price {
        font-size: 1.3rem;
        font-weight: 700;
        color: #E05674;
    }
    
    .menu-desc {
        font-size: 0.92rem;
        color: #5C5465;
        line-height: 1.5;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    .badge {
        display: inline-block;
        background: #FFF0F3;
        color: #D84062;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #FFE0E6;
    }
    
    .badge-vegan {
        background: #F0FDF4;
        color: #16A34A;
        border-color: #DCFCE7;
    }
    
    .badge-premium {
        background: #FAF5FF;
        color: #9333EA;
        border-color: #F3E8FF;
    }
    
    /* Info Banner */
    .info-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 20px 25px;
        border-left: 6px solid #FF8E9E;
        box-shadow: 0 6px 18px rgba(0,0,0,0.03);
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load Model & Index
    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"Error loading index: {exc}")
        st.stop()

    # App Header Banner
    st.markdown("""
    <div class="header-box">
        <div class="header-title">🍨 MilkLab° Gelato Dashboard</div>
        <div class="header-subtitle">ไอศกรีมเจลาโต้โฮมเมดพรีเมียม สดใหม่ วัตถุดิบนำเข้า 100%</div>
    </div>
    """, unsafe_allow_html=True)

    # Top Floating / Pop-up Action Bar
    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
    with top_col1:
        st.markdown("### 📜 เมนูไอศกรีมเจลาโต้ แนะนำประจำวัน")
    with top_col2:
        st.markdown(" ")
    with top_col3:
        if st.button("💬 คุยกับน้อง AI เจลาโต้ (Pop-up)", use_container_width=True, type="primary"):
            chatbot_popup_dialog(model, index, chunks)

    # Store Quick Info Bar
    st.markdown("""
    <div class="info-card">
        <strong>⏰ เวลาเปิดบริการ:</strong> 16:00 - 23:00 น. (เปิดทุกวันยกเว้นวันจันทร์) &nbsp;|&nbsp; 
        <strong>🚚 บริการจัดส่ง:</strong> รัศมี 5 กม. พร้อมแพ็คเจลเก็บความเย็น (ค่าส่ง 30฿) &nbsp;|&nbsp;
        <strong>🛡️ สารแพ้อาหาร:</strong> ปลอดถั่ว (Nut-Free) & ปลอดกลูเตน (Gluten-Free) ทุกเมนู
    </div>
    """, unsafe_allow_html=True)

    # Display Gelato Menu Grid (3 Columns)
    cols = st.columns(3)
    
    for idx, item in enumerate(GELATO_MENUS):
        col_target = cols[idx % 3]
        with col_target:
            # Render Badges
            badges_html = ""
            for b in item["badges"]:
                badge_cls = "badge"
                if "Vegan" in b:
                    badge_cls += " badge-vegan"
                elif "Premium" in b or "Signature" in b:
                    badge_cls += " badge-premium"
                badges_html += f'<span class="{badge_cls}">{b}</span>'

            card_html = f"""
            <div class="menu-card">
                <div class="menu-icon">{item["icon"]}</div>
                <div class="menu-name">{item["name"]}</div>
                <div class="menu-en">{item["en_name"]}</div>
                <div>{badges_html}</div>
                <div class="menu-desc">{item["desc"]}</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="menu-price">{item["price"]} บาท <span style="font-size:0.85rem; color:#9A8C98; font-weight:400;">/ {item["size"]}</span></div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Interactive Query Button for each card
            if st.button(f"❓ สอบถามเมนูนี้ ({item['name']})", key=f"btn_ask_{item['id']}", use_container_width=True):
                chatbot_popup_dialog(model, index, chunks, initial_query=item["query"])

    st.markdown("---")

    # Bottom Call-to-action bar
    st.markdown("""
    <div style="text-align: center; color: #9A8C98; padding: 20px;">
        MilkLab° Gelato Solopreneur Starter • Powered by Streamlit + FAISS + Gemini AI 🍨
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
