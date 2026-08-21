# Pivot to: UltraSmoothhh Gelato Lab

## Context
- **เจ้าของ/persona**: เกียรติภูมิ (Kiattiphum / iPxum)
- **ธุรกิจคืออะไร**: UltraSmoothhh Gelato Lab° ร้านเจลาโต้โฮมเมดคราฟต์ระดับพรีเมียม สไตล์อิตาเลียน ใช้วัตถุดิบธรรมชาติสดใหม่ 100% (นมสดฮอกไกโดแท้, ผงโกโก้ Valrhona 70%, ชาเขียวมัทฉะเกรดพิธีการจากอุจิ เกียวโต, สตรอว์เบอร์รีสด และเนื้อมะม่วงมหาชนกฉ่ำ)
- **ปัญหา solopreneur ที่จะแก้**: 
  - ระบบบริการลูกค้าแบบออโตเมชันผ่าน RAG AI ตอบคำถามเรื่องเมนู สารแพ้อาหาร (Nut-Free, Gluten-Free, Vegan) และบริการจัดส่งเจลเย็นตลอด 24 ชั่วโมง
  - ระบบสร้างแคปชั่นโปรโมตเมนูบน Social Media อัตโนมัติด้วย Gemini AI
  - ระบบบันทึกออเดอร์และยอดขายสินค้าลง Google Sheets พร้อมส่งแจ้งเตือนผ่าน Telegram/LINE ทันที

## สิ่งที่เปลี่ยนจาก MilkLab°
- **caption_generator**: ปรับเปลี่ยน Prompt Template และ persona เป็นแบรนด์ *UltraSmoothhh Gelato Lab°*
- **sales_logger**: ปรับเปลี่ยนหัวตารางบันทึกยอดขายและข้อความแจ้งเตือน Notification ใน Telegram/LINE ให้สะท้อนแบรนด์ *UltraSmoothhh Gelato Lab°*
- **agent_harness**: ปรับเปลี่ยน Tool Schema (`log_sale`, `query_sales`, `query_inventory`, `send_alert`) ให้วิเคราะห์ข้อมูลและเรียกใช้ Tool ได้ครอบคลุมสินค้าเจลาโต้คราฟต์
- **RAG kb**: สร้าง `ultrasmooth_gelato_kb.md` บันทึกข้อมูลเมนู สารก่อภูมิแพ้ FAQ และนโยบายจัดส่งเย็น 45 นาที
- **app.py**: ปรับแต่ง UI Branding เป็น UltraSmoothhh Gelato Lab°, ปรับเปลี่ยน Floating Popover AI Concierge ตอบคำถามยืดหยุ่น สนุกสนานน่ารัก และคงความถูกต้องตาม Knowledge Base

## รายการเมนู / สินค้า / service
| ชื่อเมนู (Menu Item) | ราคา (บาท) | หมายเหตุ / จุดเด่น |
|---|---|---|
| **Hokkaido Milk Gelato** (เจลาโต้นมสดฮอกไกโด) | 80 | นมสดฮอกไกโดแท้ 100% เข้มข้น หอมนุ่ม (ถ้วย 120g) |
| **Dark Chocolate Gelato** (เจลาโต้ดาร์กช็อกโกแลต) | 85 | ผงโกโก้พรีเมียมเข้มข้น 70% เข้มข้นถึงใจ (ถ้วย 120g) |
| **Strawberry Sorbet Gelato** (เจลาโต้สตรอว์เบอร์รีซอร์เบต์) | 85 | สตรอว์เบอร์รีสดแท้ 100% Vegan / Dairy-Free (ถ้วย 120g) |
| **Matcha Green Tea Gelato** (เจลาโต้ชาเขียวมัทฉะ) | 90 | ผงมัทฉะเกรดพิธีการจากอุจิ เกียวโต (ถ้วย 120g) |
| **Mahachanok Mango Sorbet** (เจลาโต้มะม่วงมหาชนกซอร์เบต์) | 80 | เนื้อมะม่วงมหาชนกสดหวานฉ่ำ Vegan / Dairy-Free (ถ้วย 120g) |
