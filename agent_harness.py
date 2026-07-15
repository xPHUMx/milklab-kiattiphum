"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log

นักศึกษาต้องเติม TODO ใน 3 จุด ใน Session 2 Lab 2.3
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from datetime import datetime
from sales_logger import append_to_sheet, send_notification


TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """ส่ง cmd ไป Gemini พร้อม TOOL_SCHEMA ขอให้ตอบเป็น JSON {tool, args}

    Returns dict {"tool": <name>, "args": <dict>}
    Raises RuntimeError ถ้า parse ไม่ได้
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    คุณเป็น AI Agent ของร้าน MilkLab หน้าที่ของคุณคือวิเคราะห์คำสั่งภาษาไทยและเลือกเครื่องมือ (Tool) ที่เหมาะสมที่สุด พร้อมสกัดข้อมูล (Arguments) สำหรับเรียกใช้งานเครื่องมือดังกล่าว

    คำสั่งจากผู้ใช้: "{cmd}"
    เวลาปัจจุบัน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    เครื่องมือที่มีให้เลือกใช้งาน:
    1. log_sale: บันทึกยอดขายลงระบบ
       - menu (string): ชื่อเมนูอาหาร/เครื่องดื่ม เช่น นมหมีฮอกไกโด, นมโกโก้บราวนี่, นมเสาวรส, นมเย็นใส่วุ้นนม
       - qty (integer): จำนวนชิ้น/ขวด/หน่วย
       - price (number): ราคาต่อหน่วย
    2. query_sales: ตรวจสอบ/ดูยอดขายตามวันที่ระบุ
       - date (string): วันที่ในรูปแบบ YYYY-MM-DD (เช่น 2026-07-15) หากผู้ใช้ระบุ "วันนี้" ให้แปลงเป็นวันที่ปัจจุบัน ({datetime.now().strftime('%Y-%m-%d')}) หากระบุ "เมื่อวาน" ให้แปลงย้อนหลัง 1 วัน
    3. send_alert: ส่งแจ้งเตือนทั่วไปเข้าบอทแชท
       - message (string): ข้อความแจ้งเตือนที่ต้องการส่ง

    วิเคราะห์และแปลงผลลัพธ์เป็นโครงสร้าง JSON ดังต่อไปนี้เท่านั้น:
    {{
        "tool": "ชื่อเครื่องมือที่เลือกใช้ (ต้องเป็นหนึ่งใน: log_sale, query_sales, send_alert)",
        "args": {{
            // พารามิเตอร์ตามเครื่องมือที่เลือก
        }}
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        data = json.loads(response.text.strip())
        
        if "tool" not in data or "args" not in data:
            raise ValueError("Missing 'tool' or 'args' keys in response.")
            
        allowed_tools = ["log_sale", "query_sales", "send_alert"]
        if data["tool"] not in allowed_tools:
            raise ValueError(f"Invalid tool selected: '{data['tool']}'. Must be one of {allowed_tools}")
            
        return data
    except Exception as exc:
        raise RuntimeError(f"Gemini parsing failed: {exc}")


def run_query_sales(date: str) -> str:
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        return "Error: GOOGLE_SHEETS_CREDENTIALS is not set in environment."

    import gspread
    from google.oauth2.service_account import Credentials

    try:
        info = json.loads(creds_json)
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
    except Exception as exc:
        return f"Error: Authentication failed: {exc}"

    sheet_name = os.environ.get("SPREADSHEET_NAME", "")
    try:
        if sheet_name.startswith("https://"):
            sh = client.open_by_url(sheet_name)
        elif len(sheet_name) == 44:
            sh = client.open_by_key(sheet_name)
        else:
            sh = client.open(sheet_name)
        wks = sh.get_worksheet(0)
    except Exception as exc:
        return f"Error: Could not open spreadsheet '{sheet_name}': {exc}"

    try:
        rows = wks.get_all_values()
        total_sales = 0.0
        count = 0
        for row in rows[1:]:
            if len(row) >= 5:
                ts = row[0]
                if ts.startswith(date):
                    try:
                        total_sales += float(row[4])
                        count += 1
                    except ValueError:
                        pass
        return f"ยอดขายวันที่ {date} มีทั้งหมด {count} รายการ รวมเป็นเงิน {total_sales} บาท"
    except Exception as exc:
        return f"Error: Failed to query sheet: {exc}"


def dispatch_tool(tool_call: dict) -> dict:
    """เรียก tool ตาม tool_call["tool"] ด้วย args จริง

    Returns: dict {"tool_output": ..., "user_output": ...}
    """
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    if tool_name == "log_sale":
        menu = args.get("menu")
        qty = int(args.get("qty", 0))
        price = float(args.get("price", 0.0))
        
        row = append_to_sheet(menu, qty, price)
        total = row["total"]
        
        try:
            provider = send_notification(f"บันทึก {menu} x{qty} = {total} บาท")
            tool_output = f"OK: row appended at {row['timestamp']}"
            user_output = f"บันทึกแล้วยอด {total} บาท"
        except Exception as exc:
            tool_output = f"OK: row appended at {row['timestamp']} (แจ้งเตือนล้มเหลว: {exc})"
            user_output = f"บันทึกแล้วยอด {total} บาท (แจ้งเตือนล้มเหลว)"
        
        return {
            "tool_output": tool_output,
            "user_output": user_output
        }
        
    elif tool_name == "query_sales":
        date = args.get("date")
        res = run_query_sales(date)
        return {
            "tool_output": res,
            "user_output": res
        }
        
    elif tool_name == "send_alert":
        message = args.get("message")
        try:
            provider = send_notification(message)
            res = f"ส่งการแจ้งเตือนสำเร็จผ่าน {provider}"
        except Exception as exc:
            res = f"ส่งการแจ้งเตือนล้มเหลว: {exc}"
        return {
            "tool_output": res,
            "user_output": res
        }
        
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")

    tool_call = parse_command(args.cmd)
    
    # Format args as: {key1: val1, key2: val2}
    args_str = ", ".join(f"{k}: {v}" for k, v in tool_call['args'].items())
    print(f"[LLM]  tool={tool_call['tool']} args={{{args_str}}}")

    result = dispatch_tool(tool_call)
    print(f"[TOOL] {tool_call['tool']} {result['tool_output']}")
    print(f"[USER] ← {result['user_output']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
