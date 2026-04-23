output "bastion_public_ip" {
  value = aws_instance.bastion.public_ip
}

output "alb_dns_name" {
  value       = aws_lb.ml_alb.dns_name
  description = "The DNS name of the ALB to access the ML prediction endpoint"
}

output "predict_url" {
  value = "http://${aws_lb.ml_alb.dns_name}/predict"
}

output "health_url" {
  value = "http://${aws_lb.ml_alb.dns_name}/health"
}

output "docs_url" {
  value = "http://${aws_lb.ml_alb.dns_name}/docs"
}

output "ml_node_private_ip" {
  value = aws_instance.ml_node.private_ip
}