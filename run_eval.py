"""MilkLab RAG Retrieval Evaluation Script (S3 Part 3).

Evaluates Precision@3, Recall@3 over 10 test cases and plots top-1 similarity histogram.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from app import load_index

# 1. Prepare 10 Ground Truth QA Test Cases
EVAL_DATASET = [
    {
        "query": "ร้านเปิดกี่โมงและปิดกี่โมง",
        "ground_truth": ["[เกี่ยวกับร้าน] MilkLab° Gelato เป็นร้านไอศกรีมเจลาโต้โฮมเมดพรีเมียม เปิดบริการทุกวันยกเว้นวันจันทร์ เวลา 16:00 ถึง 23:00 น."]
    },
    {
        "query": "เจลาโต้นมสดฮอกไกโดราคาเท่าไหร่และขนาดเท่าไหร่",
        "ground_truth": ["[เมนูเจลาโต้หลัก] เจลาโต้นมสดฮอกไกโด (Hokkaido Milk Gelato): 80 บาท (นมสดฮอกไกโดแท้ 100% เข้มข้น หอมนุ่ม) ขนาดถ้วย 120g"]
    },
    {
        "query": "เจลาโต้ดาร์กช็อกโกแลตรสชาติยังไงราคาเท่าไหร่",
        "ground_truth": ["[เมนูเจลาโต้หลัก] เจลาโต้ดาร์กช็อกโกแลต (Dark Chocolate Gelato): 85 บาท (ผงโกโก้พรีเมียมเข้มข้น 70%) ขนาดถ้วย 120g"]
    },
    {
        "query": "มีส่วนผสมของถั่วไหม ลูกค้าแพ้ถั่วกินได้ไหม",
        "ground_truth": ["[Allergen & สารก่อภูมิแพ้] ปลอดถั่ว (Nut-Free ทุกเมนู ลูกค้าแพ้ถั่วทานได้อย่างปลอดภัย)"]
    },
    {
        "query": "มีเจลาโต้รสไหนบ้างที่คนทานมังสวิรัติหรือวีแกนกินได้",
        "ground_truth": [
            "[เมนูเจลาโต้หลัก] เจลาโต้สตรอว์เบอร์รีซอร์เบต์ (Strawberry Sorbet Gelato): 85 บาท (สตรอว์เบอร์รีสดแท้ 100%) ขนาดถ้วย 120g",
            "[Allergen & สารก่อภูมิแพ้] ซอร์เบต์ (สตรอว์เบอร์รีซอร์เบต์, มะม่วงซอร์เบต์): ปลอด dairy/lactose 100% (ผู้ทานมังสวิรัติ / Vegan ทานได้)"
        ]
    },
    {
        "query": "ค่าส่งเท่าไหร่และส่งไกลแค่ไหนมีแพ็คเจลเย็นไหม",
        "ground_truth": ["[FAQ] **ส่งได้ไกลแค่ไหน**: บริการจัดส่งรัศมี 5 กม. พร้อมแพ็คเจลเก็บความเย็น ค่าส่ง 30 บาท"]
    },
    {
        "query": "เจลาโต้ชาเขียวมัทฉะใช้ชาเขียวจากไหนราคาเท่าไหร่",
        "ground_truth": ["[เมนูเจลาโต้หลัก] เจลาโต้ชาเขียวมัทฉะ (Matcha Green Tea Gelato): 90 บาท (ผงมัทฉะเกรดพิธีการจากอุจิ เกียวโต) ขนาดถ้วย 120g"]
    },
    {
        "query": "สั่งจองเจลาโต้ล่วงหน้าทางไหนสั่งได้ถึงกี่โมง",
        "ground_truth": ["[FAQ] **จองล่วงหน้าได้ไหม**: ได้ ผ่าน LINE OA สั่งก่อน 15:00 น."]
    },
    {
        "query": "การสั่งซื้อไอติมเจลาโต้มีกำหนดออเดอร์ขั้นต่ำหรือไม่",
        "ground_truth": ["[FAQ] **ออเดอร์ขั้นต่ำ**: ไม่มีออเดอร์ขั้นต่ำ"]
    },
    {
        "query": "เจลาโต้มะม่วงมหาชนกซอร์เบต์ราคาเท่าไหร่",
        "ground_truth": ["[เมนูเจลาโต้หลัก] เจลาโต้มะม่วงมหาชนกซอร์เบต์ (Mahachanok Mango Sorbet): 80 บาท (เนื้อมะม่วงมหาชนกสดหวานฉ่ำ) ขนาดถ้วย 120g"]
    }
]


def evaluate_retrieval(k: int = 3):
    model, index, chunks = load_index()

    precisions = []
    recalls = []
    top1_scores = []

    print(f"="*70)
    print(f"RAG RETRIEVAL EVALUATION (Top-k = {k})")
    print(f"="*70)

    for i, item in enumerate(EVAL_DATASET, 1):
        query = item["query"]
        gt = item["ground_truth"]

        # Encode & search index
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_emb, k)

        retrieved_chunks = [chunks[idx] for idx in indices[0] if 0 <= idx < len(chunks)]
        top1_score = float(scores[0][0])
        top1_scores.append(top1_score)

        # Calculate hits
        hits = sum(1 for c in retrieved_chunks if c in gt)
        precision = hits / k
        recall = hits / len(gt) if len(gt) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

        print(f"\n[Case {i}] Query: '{query}'")
        print(f"   Top-1 Score: {top1_score:.4f}")
        print(f"   Precision@{k}: {precision:.4f} | Recall@{k}: {recall:.4f}")
        print(f"   Hits: {hits}/{len(gt)}")
        for r_idx, r_chunk in enumerate(retrieved_chunks, 1):
            is_gt = "✓" if r_chunk in gt else " "
            print(f"   - [{r_idx}] [{is_gt}] {r_chunk}")

    mean_precision = float(np.mean(precisions))
    mean_recall = float(np.mean(recalls))

    print(f"\n"+"="*70)
    print(f"SUMMARY RESULTS:")
    print(f"   Mean Precision@{k}: {mean_precision:.4f} ({mean_precision*100:.2f}%)")
    print(f"   Mean Recall@{k}:    {mean_recall:.4f} ({mean_recall*100:.2f}%)")
    print(f"   Mean Top-1 Similarity Score: {float(np.mean(top1_scores)):.4f}")
    print(f"="*70)

    # Plot histogram of top-1 similarity scores
    plt.figure(figsize=(8, 5))
    plt.hist(top1_scores, bins=5, color='teal', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Top-1 Similarity Scores (MilkLab RAG)")
    plt.xlabel("Similarity Score (Cosine)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("top1_similarity_histogram.png")
    print("\nSaved histogram plot to 'top1_similarity_histogram.png'")

    return mean_precision, mean_recall, top1_scores


if __name__ == "__main__":
    evaluate_retrieval(k=3)
