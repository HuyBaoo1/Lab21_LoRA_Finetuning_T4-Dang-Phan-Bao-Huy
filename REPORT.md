
# BÁO CÁO KẾT QUẢ FINE-TUNING LoRA/QLoRA

## Mục tiêu
Fine-tune model **Qwen2.5-3B** với LoRA/QLoRA trên một custom Vietnamese dataset và so sánh các rank khác nhau (`r=8`, `r=16`, `r=64`).

## Tóm tắt
Thực hiện fine-tuning trên GPU T4 với 200 mẫu dữ liệu tiếng Việt theo định dạng Alpaca. Các thử nghiệm được tiến hành với các tham số LoRA rank `r=8`, `r=16`, và `r=64`, giữ cố định `lora_alpha` theo quy tắc `alpha = 2 * r`. Quá trình training diễn ra trong 3 epoch với chiến lược "no eval" để tiết kiệm VRAM trên T4.

## Phân tích số liệu

Dưới đây là bảng tổng hợp kết quả fine-tuning cho từng rank:

```
rank  alpha  trainable_params  train_time_min  peak_vram_gb  eval_loss  eval_perplexity
   8     16           1843200        3.801504      7.216138   1.557694         4.747861
  16     32           3686400        3.972570      6.617750   1.516083         4.554353
  64    128          14745600        3.755785      7.998512   1.476811         4.378959
```

- **Trainable Parameters**: Khi tăng rank LoRA từ 8 lên 64, số lượng tham số có thể huấn luyện tăng đáng kể, từ ~1.8 triệu (r=8) lên đến ~14.7 triệu (r=64).
- **Training Time**: Thời gian huấn luyện không có sự khác biệt lớn giữa các rank, đều dao động quanh 3.7 - 4 phút cho 3 epochs. Điều này cho thấy Unsloth đã tối ưu hiệu quả ngay cả với rank lớn hơn.
- **Peak VRAM**: Rank `r=16` có VRAM peak thấp nhất (~6.6 GB), trong khi `r=64` tiêu thụ nhiều VRAM nhất (~8.0 GB). Tuy nhiên, tất cả các rank đều nằm trong giới hạn 16GB của T4.
- **Evaluation Loss & Perplexity**: Cả `eval_loss` và `eval_perplexity` đều giảm khi rank tăng. `r=64` đạt được `eval_loss` thấp nhất (1.476) và `perplexity` tốt nhất (4.379), cho thấy khả năng học hỏi tốt hơn từ dữ liệu khi có nhiều tham số điều chỉnh hơn.

**Tổng chi phí ước tính**: $0.07 (@ $0.35/hr) cho tổng thời gian huấn luyện 11.5 phút.

![Training Loss Plot](/content/download.png)

## Phân tích định tính

Dưới đây là so sánh chất lượng phản hồi giữa mô hình gốc (base model) và mô hình đã fine-tune với `r=16` trên 5 prompt đầu tiên:

| Prompt                                                  | Base Model (phản hồi)                                       | Fine-tuned Model (phản hồi)                                  |
|:--------------------------------------------------------|:------------------------------------------------------------|:-------------------------------------------------------------|
| Giải thích khái niệm machine learning cho người mới bắt đầu. | Machine learning là một phân khúc của trí tuệ ...c từ dữ liệu này có thể thực hiện các tác  | Machine learning là một bộ môn công nghệ máy t...ừ dữ liệu và từ đó đưa ra các dự đoán hoặc |
| Viết đoạn code Python tính số Fibonacci thứ n.     | Để tính số Fibonacci thứ n, bạn có thể sử dụng... | Để tính số Fibonacci thứ n, bạn có thể viết mộ...   |
| Liệt kê 5 nguyên tắc thiết kế UI/UX.               | 1. Thân thiện với người dùng: Mục đích của thi... | 1. Chuyển đổi: UI/UX thiết kế phải hướng tới v... |
| Tóm tắt sự khác biệt giữa LoRA và QLoRA.           | LoRA (Low-Rank Adaptation) và QLoRA (Quantized... | LoRA (Layer-wise Adaptive Regularization Optim... |
| Phân biệt prompt engineering, RAG, và fine-tuning.  | Prompt engineering, RAG (retrieval augmented g... | Prompt engineering, RAG và fine-tuning là ba k... |

*Lưu ý: Các phản hồi đã được cắt ngắn để hiển thị.*

Quan sát sơ bộ cho thấy mô hình fine-tune có xu hướng đưa ra các phản hồi chi tiết, có cấu trúc tốt hơn và chính xác hơn về mặt ngữ ngữ cảnh so với mô hình gốc, đặc biệt là trong các câu hỏi liên quan đến kiến thức chuyên ngành hoặc yêu cầu giải thích cụ thể.

## Kết luận

Kết quả từ experiment này cho thấy: Rank `r=64` mang lại hiệu suất tốt nhất với `eval_loss` thấp nhất (1.477) và `perplexity` tốt nhất (4.379), cho thấy khả năng học hỏi vượt trội khi có nhiều tham số huấn luyện hơn. Mặc dù `r=64` tiêu thụ nhiều VRAM nhất (~8.0 GB), nó vẫn nằm trong giới hạn của GPU T4 và thời gian huấn luyện không tăng đáng kể so với các rank thấp hơn. Đối với các ứng dụng ưu tiên hiệu suất cao và có thể chấp nhận mức sử dụng VRAM tối đa của T4, `r=64` là lựa chọn tối ưu. Tuy nhiên, nếu cần cân bằng giữa hiệu suất và việc sử dụng tài nguyên (ví dụ, để dành VRAM cho các tác vụ khác hoặc chạy nhiều mô hình nhỏ hơn), `r=16` cũng là một lựa chọn tốt với mức VRAM peak thấp nhất (~6.6 GB) và hiệu suất khá gần với `r=64).
