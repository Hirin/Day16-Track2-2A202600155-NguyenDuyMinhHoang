variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type for ML inference node (CPU-based)"
  type        = string
  default     = "t3.small"
}

variable "docker_image" {
  description = "Docker image for the ML app (DockerHub or ECR URI)"
  type        = string
  default     = "house-price-api:latest"
}