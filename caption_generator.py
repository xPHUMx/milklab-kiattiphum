"""UltraSmoothhh Gelato Caption Generator (S1).

Usage:
    python caption_generator.py

Reads GOOGLE_API_KEY from env. Generates a Thai social media caption for UltraSmoothhh Gelato Lab° menu items.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


PROMPT_TEMPLATE = """\
คุณคือ Social Media Manager ของแบรนด์ไอศกรีมเจลาโต้คราฟต์พรีเมียม "UltraSmoothhh Gelato Lab°"

จงเขียนแคปชั่นภาษาไทยที่น่ารับประทาน ชวนหิวยามดึก สำหรับโปรโมตเมนูเจลาโต้: {menu}

เงื่อนไข:
- โทนสนุกสนาน อบอุ่น ชวนชิม นุ่มละมุน ใช้คำง่าย มีชีวิตชีวา และใส่ emoji 🍨✨
- เน้นจุดเด่นวัตถุดิบธรรมชาติสดใหม่ 100% สไตล์อิตาเลียนโฮมเมด
- ต้องมี Call-to-Action ปิดท้าย เช่น "สั่งเลขน้อง ultrasmoothhh ได้เลยน้าา 🍨" หรือ "ทักแชทสั่ง Delivery พร้อมเจลเย็น 45 นาที!"
- ห้ามใช้ em dash
"""


def generate_caption(menu: str, api_key: str | None = None) -> str:
    """Generate a Thai caption for the given milk menu item."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT_TEMPLATE.format(menu=menu),
    )
    return response.text or ""


def main() -> int:
    load_dotenv()
    menu = input("เมนูที่จะโปรโมต: ").strip()
    if not menu:
        print("กรุณาใส่ชื่อเมนู")
        return 1
    caption = generate_caption(menu)
    print()
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
