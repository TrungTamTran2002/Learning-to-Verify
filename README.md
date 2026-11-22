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

Hai lượt thí nghiệm thử (pilot) đều là DPO thường ở mức cả lời giải, gọi là **Sol-DPO**, **không phải** USC-DPO (chưa có token trạng thái, cắt KV cache, cặp mức bước và cặp kiểm chứng).

- **Run A (10/2026):** 81 cặp, 50 câu test. Ba mô hình ngang nhau (37/50).
- **Run B (11/2026):** SFT-CoT huấn luyện trên 7.393 mẫu, lr 1e-5; 146 cặp; 100 câu test.
  - Chấm strict: Base 14%, SFT-CoT 65%, Sol-DPO 68% (Sol-DPO vs SFT-CoT p = 0,45).
  - Chấm mù theo nội dung (v2): Base 74%, SFT-CoT 66%, Sol-DPO 68%. Chênh lệch kiểu strict chủ yếu do định dạng đáp án.
  - DPO vẫn dùng lr 1e-6: thay đổi trọng số khoảng 1e-4, loss ≈ ln 2, 68/100 lời giải giống hệt SFT-CoT.
  - 24/25 cặp reasoning-correction gán nhầm lời giải đúng là sai (bộ kiểm tra số học đọc sai "x", "$", dấu phẩy).

Chi tiết và kế hoạch đánh giá USC-DPO đầy đủ: `paper_usc_dpo.pdf` (Mục 5–8). Nhãn chấm lại: `analysis/regrade_labels.csv` (Run A), `analysis/regrade_labels_runB.csv` (Run B).
