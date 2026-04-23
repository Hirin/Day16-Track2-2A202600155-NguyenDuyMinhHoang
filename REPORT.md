# Lab 16 — Báo cáo triển khai ML Endpoint trên AWS

## 1. API Predict thành công

![curl predict](curl.png)

## 2. Swagger UI

![swagger docs](docs.png)

## 3. API Response chi tiết

![api response](api.png)

## 4. AWS Billing - Billing bị trùng với con chatbot Discord e host nên không biết filter ra sao :)) 

![billing dashboard](billing.png)

## 5. Cold Start Time

| Giai đoạn | Thời gian |
|-----------|-----------|
| `terraform apply` (tạo hạ tầng) | ~2.5 phút |
| `user_data.sh` (cài Docker + build image + train model) | ~3.5 phút |
| **Tổng** | **~6 phút** |

## 6. Thông tin triển khai

| Mục | Giá trị |
|-----|---------|
| Instance type | `t3.small` (CPU) |
| AMI | Ubuntu 22.04 |
| Model | GradientBoostingRegressor (sklearn) |
| Dataset | California Housing |
| Serving | FastAPI + uvicorn (Docker) |
| Region | us-east-1 |
| ALB endpoint | `ml-inference-alb-c252abaa-32883847.us-east-1.elb.amazonaws.com` |
