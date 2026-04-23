# Hướng dẫn Thực hành LAB 16: Cloud ML Environment Setup (2.5h)

Chào mừng các bạn đến với Lab 16. Trong bài thực hành này, chúng ta sẽ thiết lập một môi trường Cloud ML hoàn chỉnh trên AWS bằng cách sử dụng **Terraform** (Infrastructure as Code) và **Docker**.

Mục tiêu của bài lab là huấn luyện và triển khai một mô hình **Machine Learning dự đoán giá nhà** (House Price Prediction) sử dụng dataset California Housing. Mô hình sẽ được đóng gói trong Docker container, chạy trên một EC2 instance **CPU** (không yêu cầu GPU), và cung cấp REST API thông qua Load Balancer.

> **Lưu ý:** Phiên bản này không yêu cầu GPU, phù hợp với tất cả các tài khoản AWS kể cả khi chưa được cấp quota GPU.

---

## Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    ML-VPC (10.0.0.0/16)              │   │
│  │                                                      │   │
│  │   ┌─────────────┐         ┌───────────────────────┐  │   │
│  │   │ Public      │         │ Private Subnet        │  │   │
│  │   │ Subnet      │         │                       │  │   │
│  │   │             │  SSH    │  ┌──────────────────┐  │  │   │
│  │   │  Bastion ───┼────────►│  │ ML Node (t3.small)│ │  │   │
│  │   │  Host       │         │  │                  │  │  │   │
│  │   │             │         │  │  Docker           │  │  │   │
│  │   └─────────────┘         │  │  └─ FastAPI       │  │  │   │
│  │                           │  │     └─ sklearn    │  │  │   │
│  │   ┌─────────────┐  :8000 │  │        model      │  │  │   │
│  │   │  ALB (:80) ─┼────────►  └──────────────────┘  │  │   │
│  │   └─────────────┘         │         ▲ NAT GW      │  │   │
│  │         ▲                 └─────────┼─────────────┘  │   │
│  └─────────┼────────────────────────────────────────────┘   │
│            │                                                │
└────────────┼────────────────────────────────────────────────┘
             │
      curl /predict
        (Internet)
```

---

## Phần 1: Chuẩn bị tài khoản AWS và thiết lập IAM (Least-Privilege)

Để làm việc với AWS an toàn, chúng ta không bao giờ sử dụng tài khoản Root. Thay vào đó, bạn sẽ tạo một IAM User thuộc một IAM Group với các quyền vừa đủ (least-privilege) để Terraform có thể triển khai hạ tầng.

### Bước 1.1: Truy cập AWS Console

1. Đăng nhập vào [AWS Management Console](https://console.aws.amazon.com/) bằng tài khoản Root hoặc tài khoản Admin của bạn.
2. Trên thanh tìm kiếm, gõ **IAM** và chọn dịch vụ **IAM (Identity and Access Management)**.

### Bước 1.2: Tạo IAM Group và gắn quyền (Policies)

1. Trong menu bên trái của IAM, chọn **User groups** -> click **Create group**.
2. Đặt tên nhóm: `ML-Lab-Group`.
3. Trong phần **Attach permissions policies**, bạn cần tìm và tick chọn các quyền (roles) sau. **Giải thích tại sao cần:**
   - `AmazonEC2FullAccess`: Cần thiết để Terraform tạo máy chủ ảo (Bastion Host, ML Node), Key Pairs, và Security Groups.
   - `AmazonVPCFullAccess`: Cần thiết để Terraform tạo môi trường mạng (VPC, Subnets, Internet Gateway, NAT Gateway, Route Tables).
   - `ElasticLoadBalancingFullAccess`: Cần thiết để tạo Application Load Balancer (ALB) giúp phân phối traffic từ internet vào private ML Node.
4. Click **Create user group**.

> **Lưu ý:** So với phiên bản GPU, chúng ta **không cần** `IAMFullAccess` vì ML Node không cần IAM Role/Instance Profile.

### Bước 1.3: Tạo IAM User và lấy Access Keys

1. Trong menu bên trái, chọn **Users** -> click **Create user**.
2. Đặt tên user: `ml-lab-user`. Click Next.
3. Chọn **Add user to group**, tick chọn nhóm `ML-Lab-Group` vừa tạo. Click Next -> **Create user**.
4. Bấm vào tên user `ml-lab-user` vừa tạo. Chuyển sang tab **Security credentials**.
5. Kéo xuống phần **Access keys**, click **Create access key**.
6. Chọn **Command Line Interface (CLI)** -> Check đồng ý -> Next -> **Create access key**.
7. **LƯU Ý:** Copy `Access key ID` và `Secret access key` lưu vào nơi an toàn. Bạn sẽ không thể xem lại Secret key sau khi đóng cửa sổ này.

> **⚠️ Bảo mật:** Tuyệt đối **KHÔNG** commit Access Key vào Git hay dán lên bất kỳ đâu công khai. Nếu key bị lộ, hãy vào IAM Console xóa key cũ và tạo key mới ngay lập tức.

---

## Phần 2: Cài đặt và cấu hình môi trường Local

Trên máy tính cá nhân của bạn, mở Terminal/Command Prompt.

### Bước 2.1: Cấu hình AWS CLI

Đảm bảo bạn đã cài đặt [AWS CLI](https://aws.amazon.com/cli/). Gõ lệnh sau để cấu hình tài khoản vừa tạo:

```bash
aws configure
```

Nhập các thông tin:

- **AWS Access Key ID**: (Dán Access key ID của bạn)
- **AWS Secret Access Key**: (Dán Secret access key của bạn)
- **Default region name**: `us-east-1` (Bắt buộc dùng us-east-1 cho lab này)
- **Default output format**: `json`

### Bước 2.2: Tạo SSH Key Pair

Tạo cặp khóa SSH để truy cập vào các máy chủ EC2:

```bash
cd terraform
ssh-keygen -t rsa -b 4096 -f lab-key -N ""
```

Lệnh này sẽ tạo ra 2 file: `lab-key` (private key) và `lab-key.pub` (public key) trong thư mục `terraform/`.

---

## Phần 3: Tìm hiểu Mô hình ML

Trước khi triển khai lên Cloud, hãy hiểu mô hình mà chúng ta sẽ deploy.

### 3.1: Dataset — California Housing

Đây là dataset có sẵn trong thư viện `scikit-learn`, chứa thông tin về các block nhà ở California (Mỹ) từ cuộc điều tra dân số năm 1990. Bao gồm **8 features** đầu vào:

| Feature | Mô tả |
|---------|-------|
| `MedInc` | Thu nhập trung bình của hộ dân trong khu vực |
| `HouseAge` | Tuổi trung bình của nhà trong khu vực |
| `AveRooms` | Số phòng trung bình mỗi hộ |
| `AveBedrms` | Số phòng ngủ trung bình mỗi hộ |
| `Population` | Dân số trong khu vực |
| `AveOccup` | Số người trung bình mỗi hộ |
| `Latitude` | Vĩ độ |
| `Longitude` | Kinh độ |

**Target:** Giá nhà trung bình (đơn vị: $100,000).

### 3.2: Mô hình — GradientBoostingRegressor

Chúng ta sử dụng pipeline gồm:
1. **StandardScaler** — Chuẩn hóa features về cùng scale.
2. **GradientBoostingRegressor** (200 estimators, max_depth=5) — Mô hình ensemble mạnh mẽ cho bài toán regression.

### 3.3: Thử train trên local (Tùy chọn)

Nếu muốn thử nghiệm trước khi deploy, bạn có thể chạy:

```bash
cd ml-app
pip install -r requirements.txt
python train_model.py
```

Kết quả sẽ in ra các metrics (MAE, RMSE, R²) và lưu model vào `ml-app/artifacts/`.

---

## Phần 4: Triển khai Hạ tầng với Terraform

Terraform là công cụ giúp chúng ta khởi tạo hạ tầng AWS hoàn toàn tự động bằng code. Kiến trúc bao gồm:

- Mạng **Private VPC** cách ly hoàn toàn với bên ngoài.
- **Bastion Host** (t3.micro) ở Public Subnet: Dùng làm trạm trung chuyển an toàn nếu cần SSH vào ML Node.
- **ML Node** (t3.small — CPU) ở Private Subnet: Chạy Docker chứa FastAPI + scikit-learn model.
- **NAT Gateway**: Cho phép Private Subnet kéo Docker image và cài đặt dependencies từ internet.
- **Application Load Balancer (ALB)**: Mở cổng 80 (HTTP) để nhận API request và đẩy vào ML Node ở cổng 8000.

### Bước 4.1: Khởi tạo Terraform

Di chuyển vào thư mục code Terraform:

```bash
cd terraform
terraform init
```

### Bước 4.2: Xem trước (Plan)

Kiểm tra những gì Terraform sẽ tạo:

```bash
terraform plan
```

### Bước 4.3: Triển khai (Apply)

Chạy lệnh apply để Terraform bắt đầu tạo tài nguyên trên AWS:

```bash
terraform apply
```

Gõ `yes` khi được hỏi. Quá trình này sẽ mất khoảng **5–10 phút** (phần lớn thời gian là để khởi tạo NAT Gateway).

*Mẹo: Các bạn hãy bắt đầu bấm giờ (benchmark) từ lúc gõ `yes` ở bước này nhé!*

---

## Phần 5: Kiểm tra ML Endpoint (Inference)

Khi `terraform apply` chạy xong, màn hình terminal sẽ in ra các thông số quan trọng (Outputs). Trông sẽ giống thế này:

```text
Outputs:

alb_dns_name     = "ml-inference-alb-xxxxxx.us-east-1.elb.amazonaws.com"
bastion_public_ip = "100.x.x.x"
docs_url         = "http://ml-inference-alb-xxxxxx.us-east-1.elb.amazonaws.com/docs"
health_url       = "http://ml-inference-alb-xxxxxx.us-east-1.elb.amazonaws.com/health"
ml_node_private_ip = "10.0.1x.x"
predict_url      = "http://ml-inference-alb-xxxxxx.us-east-1.elb.amazonaws.com/predict"
```

**Quan trọng:** Mặc dù Terraform đã báo thành công, ML Node vẫn đang ngầm cài đặt Docker, build Docker image, và train model. **Bạn cần đợi thêm 3–5 phút** để service sẵn sàng.

### Bước 5.1: Kiểm tra Health Check

```bash
curl http://<THAY_BẰNG_ALB_DNS_NAME_CỦA_BẠN>/health
```

Kết quả mong đợi:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "GradientBoostingRegressor",
  "metrics": {"mae": 0.2949, "rmse": 0.4233, "r2": 0.8637}
}
```

### Bước 5.2: Gọi API dự đoán giá nhà bằng cURL

Thay thế URL của ALB bạn nhận được vào lệnh dưới đây và chạy thử:

```bash
curl -X POST http://<THAY_BẰNG_ALB_DNS_NAME_CỦA_BẠN>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322.0,
    "AveOccup": 2.556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

Kết quả mong đợi:

```json
{
  "predicted_price_100k": 4.2815,
  "predicted_price_usd": "$428,150",
  "model_type": "GradientBoostingRegressor",
  "features_received": {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322.0,
    "AveOccup": 2.556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }
}
```

### Bước 5.3: Truy cập Swagger UI (Tùy chọn)

Mở trình duyệt và truy cập:

```
http://<THAY_BẰNG_ALB_DNS_NAME_CỦA_BẠN>/docs
```

Bạn sẽ thấy giao diện Swagger tự động sinh bởi FastAPI, nơi bạn có thể thử gọi API trực tiếp trên trình duyệt.

Nếu nhận được kết quả dự đoán, chúc mừng bạn đã triển khai thành công! Hãy ghi lại tổng thời gian (Cold start time) từ lúc chạy `terraform apply` đến lúc nhận được API response đầu tiên.

---

## Phần 6: Tiêu chí nộp bài (Deliverables)

Để hoàn thành Lab 16, sinh viên cần thu thập và nộp các kết quả sau:

1. **Ảnh chụp màn hình (Screenshot) API gọi thành công:** Chụp lại lệnh curl `/predict` và kết quả dự đoán giá nhà.
2. **Ảnh chụp Swagger UI:** Chụp màn hình trang `/docs` trên trình duyệt.
3. **Ảnh chụp màn hình AWS Billing/Cost Dashboard:**
   - Vào AWS Console -> Gõ **Billing** trên thanh tìm kiếm.
   - Chụp lại màn hình thể hiện các dịch vụ đang chạy phát sinh chi phí (EC2, NAT Gateway).
4. **Report Cold Start Time:** Ghi lại tổng thời gian triển khai (Mục tiêu: < 10 phút cho instance CPU).
5. **Mã nguồn:** Nén thư mục chứa file Terraform và `ml-app/` đã chạy thành công.

---

## Phần 7: Dọn dẹp tài nguyên (CỰC KỲ QUAN TRỌNG)

EC2 và NAT Gateway tính phí theo giờ. Mặc dù `t3.small` rẻ hơn nhiều so với GPU, bạn vẫn **NÊN** xóa toàn bộ tài nguyên ngay sau khi test xong để tránh phát sinh chi phí không cần thiết.

Chạy lệnh sau trong thư mục `terraform`:

```bash
terraform destroy
```

Gõ `yes` khi được hỏi. Quá trình xóa sẽ mất khoảng 5 phút. Hãy đợi đến khi terminal báo `Destroy complete!` để chắc chắn mọi thứ đã bị xóa.

---

## So sánh: Phiên bản GPU (LLM) vs CPU (ML)

| Tiêu chí | GPU (LLM) | CPU (ML) — Bài này |
|-----------|-----------|---------------------|
| Model | google/gemma-4-E2B-it (~GB) | GradientBoosting (~KB) |
| Instance | g4dn.xlarge (~$0.53/h) | t3.small (~$0.02/h) |
| GPU quota | Cần xin quota (có thể bị từ chối) | Không cần |
| Cold start | ~15–20 phút | ~5–10 phút |
| Chi phí 1h lab | ~$0.60+ | ~$0.10 |
| Serving framework | vLLM | FastAPI + sklearn |