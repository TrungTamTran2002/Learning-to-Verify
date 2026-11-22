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
| `DuAncntt.ipynb` | Notebook: tạo dữ liệu, huấn luyện SFT-CoT và Sol-DPO (DPO mức lời giải), đánh giá |
| `analysis/pilot_diagnostics.py` | Tái tạo các số liệu thí nghiệm thử trong bài báo (chấm lại, McNemar, độ lớn cập nhật LoRA, cỡ mẫu) |
| `analysis/regrade_labels.csv` | Nhãn đúng/sai của 150 lời giải theo 3 cách chấm |
| `USC_DPO_training_artifacts/` | Adapter DPO (LoRA), lịch sử loss/margin, 81 cặp preference, log kiểm toán |
| `USC_DPO_final_evaluation (2)/` | Kết quả đánh giá: summary, CSV, 150 dự đoán, adapter SFT và DPO |

## Tóm tắt phương pháp

Mô hình nền: **Qwen2.5-Math-1.5B**, fine-tune bằng **LoRA** trên GPU Tesla T4.

Quy trình: thử mô hình gốc → huấn luyện SFT-CoT → sinh 320 lời giải ứng viên (80 câu × 4) → gán nhãn đúng/sai → tạo 81 cặp preference → huấn luyện DPO (3 epoch, 243 bước) → đánh giá trên 50 câu GSM8K.

## Trạng thái kết quả

Thí nghiệm hiện tại là **thí nghiệm thử (pilot)** với DPO thường trên 81 cặp ở mức cả lời giải, gọi là **Sol-DPO**. Nó **không phải** USC-DPO: chưa có token trạng thái `<|step_valid|>` / `<|step_invalid|>`, chưa cắt KV cache, chưa có cặp mức bước và cặp kiểm chứng.

Kết quả chẩn đoán:
- Adapter gần như không thay đổi mô hình: thay đổi tương đối của trọng số khoảng 1e-4 (Sol-DPO) và 2e-5 (SFT-CoT), do learning rate 1e-6 / 2e-6 quá nhỏ cho LoRA.
- 33/50 lời giải giống hệt mô hình gốc từng ký tự.
- Độ chính xác ngang nhau: 37/50 ở cả ba mô hình theo bộ chấm tự động mù (v2); 50 câu chỉ phát hiện được chênh lệch khoảng 12 điểm.

Kế hoạch đánh giá USC-DPO đầy đủ (cấu hình, cỡ mẫu, ablation) nằm ở Mục 8 của `paper_usc_dpo.pdf`.
