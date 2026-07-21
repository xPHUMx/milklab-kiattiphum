"""MilkLab Sales Logger (S2).

Usage:
    python sales_logger.py --menu "นมหมีฮอกไกโด" --qty 2 --price 65

Reads GOOGLE_SHEETS_CREDENTIALS and TELEGRAM_BOT_TOKEN (or LINE_CHANNEL_TOKEN) from env.
Appends row [timestamp, menu, qty, price, total] to a Google Sheet,
then sends a notification via Telegram or LINE bot.

นักศึกษาต้องเติม TODO ใน 4 จุดด้านล่างใน Session 2 Lab 1.3
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def append_to_sheet(menu: str, qty: int, price: float) -> dict:
    """ใช้ gspread เปิด Sheet ของตัวเอง แล้ว append_row ด้วย [timestamp, menu, qty, price, total]

    Returns dict {timestamp, menu, qty, price, total} ที่ append แล้ว
    Raises RuntimeError ถ้า credentials ไม่มี หรือ Sheet ไม่ accessible
    """
    if qty <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")

    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS is not set in environment.")

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
        raise RuntimeError(f"Failed to authenticate with Google Sheets: {exc}")

    sheet_name = os.environ.get("SPREADSHEET_NAME", "milklab-sales")
    try:
        if sheet_name.startswith("https://"):
            sh = client.open_by_url(sheet_name)
        elif len(sheet_name) == 44:  # typical spreadsheet key length
            sh = client.open_by_key(sheet_name)
        else:
            sh = client.open(sheet_name)
        wks = sh.get_worksheet(0)
    except Exception as exc:
        raise RuntimeError(f"Could not open spreadsheet '{sheet_name}': {exc}")

    try:
        # Check if the worksheet is empty (or has no headers)
        first_row = wks.row_values(1)
        if not first_row:
            headers = ["Timestamp", "เมนู", "จำนวน", "ราคาต่อหน่วย (บาท)", "ยอดรวม (บาท)"]
            wks.append_row(headers)
            
            # Format headers: Slate Blue background, white bold text, centered
            wks.format("A1:E1", {
                "backgroundColor": {"red": 0.15, "green": 0.23, "blue": 0.36},
                "horizontalAlignment": "CENTER",
                "textFormat": {
                    "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                    "fontSize": 11,
                    "bold": True
                }
            })
            
            # Format columns data alignment and number formatting
            wks.format("A2:A1000", {"horizontalAlignment": "CENTER"})
            wks.format("B2:B1000", {"horizontalAlignment": "LEFT"})
            wks.format("C2:C1000", {"horizontalAlignment": "CENTER"})
            wks.format("D2:D1000", {
                "horizontalAlignment": "RIGHT",
                "numberFormat": {"type": "NUMBER", "pattern": '#,##0.00" บาท"'}
            })
            wks.format("E2:E1000", {
                "horizontalAlignment": "RIGHT",
                "textFormat": {"bold": True},
                "numberFormat": {"type": "NUMBER", "pattern": '#,##0.00" บาท"'}
            })
            
            # Freeze row 1 so it stays at the top when scrolling
            wks.freeze(rows=1)
    except Exception as exc:
        # Prevent formatting errors from failing the append if they occur
        pass

    # Format timestamp to ISO 8601 with timezone (e.g. 2026-06-25T20:13:45+07)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    if timestamp.endswith("00") and (timestamp[-5] in ["+", "-"]):
        timestamp = timestamp[:-2]

    total = float(qty * price)
    row_data = [timestamp, menu, qty, price, total]

    try:
        wks.append_row(row_data)
    except Exception as exc:
        raise RuntimeError(f"Failed to append row to worksheet: {exc}")

    return {
        "timestamp": timestamp,
        "menu": menu,
        "qty": qty,
        "price": price,
        "total": total
    }



def send_notification(message: str) -> str:
    """ส่ง message ไปยัง Telegram bot (ใช้ TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
    หรือ LINE bot (ใช้ LINE_CHANNEL_TOKEN) เลือกตัวใดตัวหนึ่ง

    Returns: provider name ที่ใช้ ("telegram" หรือ "line")
    Raises RuntimeError ถ้า no credentials หรือส่งไม่สำเร็จ
    """
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    line_token = os.environ.get("LINE_CHANNEL_TOKEN")

    if telegram_token and telegram_chat_id:
        try:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            payload = {
                "chat_id": telegram_chat_id,
                "text": message
            }
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return "telegram"
        except Exception as exc:
            raise RuntimeError(f"Telegram notification failed: {exc}")
            
    elif line_token:
        try:
            headers = {
                "Authorization": f"Bearer {line_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
            resp = requests.post("https://api.line.me/v2/bot/message/broadcast", json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            return "line"
        except Exception as exc:
            raise RuntimeError(f"LINE notification failed: {exc}")
            
    else:
        raise RuntimeError("No credentials found for Telegram or LINE in environment.")


def main() -> int:
    parser = argparse.ArgumentParser(description="MilkLab Sales Logger")
    parser.add_argument("--menu", required=True, help="ชื่อเมนู")
    parser.add_argument("--qty", type=int, required=True, help="จำนวนขวด")
    parser.add_argument("--price", type=float, required=True, help="ราคาต่อขวด")
    args = parser.parse_args()

    try:
        # TODO 3: เรียก append_to_sheet แล้ว extract total
        row = append_to_sheet(args.menu, args.qty, args.price)
        total = row["total"]
    except Exception as exc:
        print(f"[ERROR] บันทึก Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS และ share Sheet กับ service account email", file=sys.stderr)
        return 1

    try:
        # TODO 4: เรียก send_notification ด้วย message ที่บอกยอดที่บันทึก
        provider = send_notification(f"บันทึก {args.menu} x{args.qty} = {total} บาท")
    except Exception as exc:
        print(f"[WARN] บันทึก Sheet สำเร็จแต่ส่งแจ้งเตือนล้มเหลว: {exc}", file=sys.stderr)
        return 0

    print(f"[OK] บันทึกและแจ้งเตือนผ่าน {provider} เรียบร้อย ยอด {total} บาท")
    return 0


if __name__ == "__main__":
    sys.exit(main())
