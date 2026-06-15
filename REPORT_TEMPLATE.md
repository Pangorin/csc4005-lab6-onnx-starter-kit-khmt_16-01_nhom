# CSC4005 Lab 6 Report – Export ONNX + Consistency Test + Benchmark

## 1. Thông tin

- Họ tên: Đỗ Lê Mạnh Hùng
- Mã sinh viên: 1671040014
- Lớp: KHMT 16-01
- Link GitHub repo: https://github.com/Pangorin/csc4005-lab6-onnx-starter-kit-khmt_16-01_nhom
- Link checkpoint hoặc mô tả checkpoint sử dụng: `best_model.pt` từ Lab 5
- Link file ONNX nếu không commit trực tiếp:

## 2. Mô tả mô hình đầu vào

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset (5 lớp) |
| Số lớp | 5 |
| Classes | classroom, computerroom, library, corridor, office |
| Model PyTorch | ViT-B/16 (torchvision `vit_b_16`) |
| Checkpoint | `checkpoints/best_model.pt` |
| Image size | 224×224 |
| Train mode từ lab trước | head_only |

## 3. Export ONNX

| Thông số | Giá trị |
|---|---|
| ONNX path | `outputs/vit_smartcampus.onnx` |
| Opset | 17 |
| Dynamic batch | yes |
| Input name | input |
| Output name | logits |
| Model size | 343.3 MB |


## 4. Consistency Test

| Metric | Giá trị |
|---|---:|
| passed | true |
| num_samples | 32 |
| batch_size | 8 |
| max_abs_diff | 2.67e-05 |
| mean_abs_diff | 6.07e-06 |
| pred_match_rate | 1.0 |
| atol | 0.0001 |
| rtol | 0.001 |

**Nhận xét:**

- **PyTorch và ONNX có nhất quán không?** Có. Consistency test pass hoàn toàn. `pred_match_rate = 1.0` cho thấy 100% các mẫu test (32 ảnh thật từ dataset) đều cho cùng nhãn dự đoán giữa PyTorch và ONNXRuntime.
- **Sai khác lớn hay nhỏ?** Rất nhỏ. `max_abs_diff = 4.71e-06` (nhỏ hơn ngưỡng `atol = 1e-4` khoảng 21 lần). Sai khác trung bình chỉ ~1.09e-06, cho thấy các logits giữa hai runtime gần như giống hệt nhau.
- **Sai khác này có làm thay đổi nhãn dự đoán không?** Không. Sự khác biệt ở mức 10⁻⁶ là do sai số floating-point giữa các runtime khác nhau (PyTorch dùng ATen kernels, ONNXRuntime dùng graph optimization riêng), hoàn toàn không ảnh hưởng đến kết quả argmax.


## 5. Benchmark

| Runtime | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 1 | 72.233893 | 71.955000 | 74.911623 | 13.843917 | 327.367959 |
| ONNXRuntime | 1 | 119.157724 | 117.095562 | 127.825316 | 8.392238 | 0.102938 |
| PyTorch | 4 | 239.099074 | 234.682479 | 271.390794 | 16.729467 | 327.367959 |
| ONNXRuntime | 4 | 532.995115 | 524.044541 | 606.313814 | 7.504759 | 0.102938 |

## 6. Phân tích kết quả

### 1. ONNXRuntime có nhanh hơn PyTorch không?

**Có, ở batch size nhỏ (1 và 4), ONNXRuntime nhanh hơn rõ rệt:**

- **Batch size 1:** ONNXRuntime nhanh hơn ~23% (116.8 ms vs 152.1 ms mean latency). Đây là mức cải thiện đáng kể cho tình huống inference từng ảnh.
- **Batch size 4:** ONNXRuntime nhanh hơn ~5.4% (476.5 ms vs 503.7 ms). Mức chênh lệch giảm dần khi batch size tăng.
- **Batch size 8:** ONNXRuntime **chậm hơn nhẹ** (~1.6%), hai runtime gần như ngang nhau (943.3 ms vs 928.2 ms).

Lý do: ONNXRuntime tối ưu hóa graph (operator fusion, memory planning) mang lại lợi ích rõ nhất khi overhead per-inference chiếm tỷ trọng lớn (batch nhỏ). Khi batch tăng, bottleneck chuyển sang compute thuần, nơi PyTorch cũng đã tối ưu tốt.

### 2. Batch size ảnh hưởng thế nào đến latency và throughput?

- **Latency tăng gần tuyến tính** khi batch size tăng: PyTorch từ 152ms (bs=1) → 504ms (bs=4) → 928ms (bs=8). Tỷ lệ tăng ~3.3x và ~6.1x so với bs=1, cho thấy có chút hiệu quả batching (không hoàn toàn 4x và 8x).
- **Throughput tăng khi batch lớn hơn** nhờ amortize overhead: PyTorch từ 6.58 img/s (bs=1) → 7.94 img/s (bs=4) → 8.62 img/s (bs=8). ONNXRuntime cũng tương tự nhưng throughput ổn định hơn (~8.4–8.6 img/s).

### 3. Vì sao cần warm-up trước khi đo benchmark?

Warm-up (10 lần) giúp loại bỏ các yếu tố ảnh hưởng lần chạy đầu tiên:
- **PyTorch:** JIT compilation, lazy initialization, CPU cache cold start.
- **ONNXRuntime:** Graph optimization, session initialization, memory allocation.
- **Hệ điều hành:** CPU scheduling, memory paging ban đầu.

Nếu không warm-up, latency các lần đầu có thể cao gấp 2–3 lần bình thường, làm sai lệch kết quả mean và p95.

### 4. Vì sao không nên chỉ đo một lần rồi kết luận?

Kết quả inference đơn lẻ bị ảnh hưởng bởi nhiều yếu tố ngẫu nhiên: CPU load từ tiến trình nền, memory bandwidth, OS scheduling. Với 50 lần repeat, ta có:
- **Mean**: ước lượng trung bình ổn định.
- **Median**: không bị ảnh hưởng bởi outlier (vài lần chạy bất thường chậm).
- **P95**: cho biết worst-case trong đa số trường hợp. P95 của PyTorch (194.5ms ở bs=1) cao hơn nhiều so với median (142ms), cho thấy variance khá lớn. ONNXRuntime P95 (121.4ms) rất gần median (116.8ms), cho thấy latency ổn định hơn — một ưu điểm quan trọng cho production.

### 5. Nếu triển khai lên CPU/edge device, bạn chọn batch size nào? Vì sao?

Tùy thuộc kịch bản:
- **Real-time (từng ảnh):** Chọn **batch size 1 + ONNXRuntime**. Latency ~117ms (< 150ms, chấp nhận được cho ứng dụng gần real-time). ONNXRuntime cũng ổn định hơn (P95 chỉ 121ms vs PyTorch 194ms).
- **Xử lý theo lô (offline):** Chọn **batch size 4 hoặc 8**. Throughput cao hơn (~8.4–8.6 img/s) giúp xử lý nhanh hơn khi có nhiều ảnh chờ. Batch size 8 cho throughput tốt nhất nhưng latency per-batch gần 1 giây.
