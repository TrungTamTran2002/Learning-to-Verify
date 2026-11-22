# Learning to Verify: Self-Verification Training for Small Reasoning Models

Đồ án Công nghệ thông tin — nghiên cứu huấn luyện mô hình ngôn ngữ nhỏ (SLM) biết tự kiểm chứng và tự sửa lời giải toán, theo khung **USC-DPO** (Unified Step-Filtered Contrastive Direct Preference Optimization).

- **Sinh viên:** Trần Trung Tâm (520H0678), Vũ Minh Quân (522H0172)
- **Giảng viên hướng dẫn:** Thầy Vũ Hồng
- **Khoa Công nghệ thông tin, Đại học Tôn Đức Thắng**

## Nội dung kho

| Thư mục / tệp | Mô tả |
|---|---|
| `paper_usc_dpo.tex`, `paper_usc_dpo.pdf` | Bài báo (LaTeX và bản biên dịch) |
| `520H0678_TranTrungTam*.pdf` | Các bản bài báo nộp trước đó |
| `slide_giua_ky_de_thuyet_trinh.pptx` | Slide thuyết trình giữa kỳ (bản mới, dễ đọc) |
| `slide_thuyet_trinh_giua_ky.pptx` | Slide thuyết trình (bản trước) |
| `bao_cao_thuc_nghiem_tuan_nay.docx` | Báo cáo tiến độ thực nghiệm |
| `DuAncntt.ipynb` | Notebook: tạo dữ liệu, huấn luyện SFT-CoT và USC-DPO (DPO), đánh giá |
| `USC_DPO_training_artifacts/` | Adapter DPO (LoRA), lịch sử loss/margin, 81 cặp preference, log kiểm toán |
| `USC_DPO_final_evaluation (2)/` | Kết quả đánh giá: summary, CSV, 150 dự đoán, adapter SFT và DPO |

## Tóm tắt phương pháp

Mô hình nền: **Qwen2.5-Math-1.5B**, fine-tune bằng **LoRA** trên GPU Tesla T4.

Quy trình: thử mô hình gốc → huấn luyện SFT-CoT → sinh 320 lời giải ứng viên (80 câu × 4) → gán nhãn đúng/sai → tạo 81 cặp preference → huấn luyện DPO (3 epoch, 243 bước) → đánh giá trên 50 câu GSM8K.

## Trạng thái kết quả

Đây là phiên bản **rút gọn** của thiết kế đầy đủ. Các cơ chế cốt lõi (token trạng thái `<|step_valid|>` / `<|step_invalid|>`, Hard Context Truncation, số hạng phạt λ, cặp preference mức bước) **chưa được cài đặt**. Trên 50 câu GSM8K, USC-DPO rút gọn chưa cho thấy khác biệt so với Base và SFT-CoT; loss DPO vẫn gần `ln 2 ≈ 0.693`. Chi tiết trong `paper_usc_dpo.pdf`.
