variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "hf_token" {
  description = "Hugging Face Token (not needed for CPU/LightGBM, set to dummy)"
  type        = string
  sensitive   = true
  default     = "dummy"
}

variable "model_id" {
  description = "Model identifier (LightGBM for CPU mode)"
  type        = string
  default     = "lightgbm-creditcard-fraud"
}