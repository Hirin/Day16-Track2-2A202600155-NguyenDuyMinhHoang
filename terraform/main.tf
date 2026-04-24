data "aws_availability_zones" "available" {
  state = "available"
}

# 1. VPC & Subnets
resource "aws_vpc" "ml_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = { Name = "ML-VPC" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.ml_vpc.id
  cidr_block              = cidrsubnet(aws_vpc.ml_vpc.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "Public-Subnet-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.ml_vpc.cidr_block, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "Private-Subnet-${count.index}" }
}

# 2. Gateways & Routing
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.ml_vpc.id
  tags = { Name = "ML-IGW" }
}

resource "aws_eip" "nat_eip" {
  domain = "vpc"
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public[0].id
  tags = { Name = "ML-NAT" }
  depends_on    = [aws_internet_gateway.igw]
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.ml_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "public_assoc" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table" "private_rt" {
  vpc_id = aws_vpc.ml_vpc.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
}

resource "aws_route_table_association" "private_assoc" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private_rt.id
}

# 3. Security Groups
resource "aws_security_group" "alb_sg" {
  name        = "ml-alb-sg"
  description = "Allow HTTP inbound to ALB"
  vpc_id      = aws_vpc.ml_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "bastion_sg" {
  name        = "ml-bastion-sg"
  description = "Allow SSH inbound to Bastion"
  vpc_id      = aws_vpc.ml_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] 
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ml_sg" {
  name        = "ml-inference-node-sg"
  description = "Allow SSH from Bastion and HTTP from ALB"
  vpc_id      = aws_vpc.ml_vpc.id

  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion_sg.id]
  }

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 4. Key Pair & Bastion
resource "aws_key_pair" "lab_key" {
  key_name   = "ml-lab-key-${random_id.id.hex}"
  public_key = file("${path.module}/lab-key.pub")
}

resource "random_id" "id" {
  byte_length = 4
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "bastion" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = "t3.micro"
  subnet_id                   = aws_subnet.public[0].id
  vpc_security_group_ids      = [aws_security_group.bastion_sg.id]
  key_name                    = aws_key_pair.lab_key.key_name
  associate_public_ip_address = true
  tags = { Name = "ML-Bastion-Host" }
}

# 5. CPU ML Instance (thay thế GPU Node theo Phần 7 README)
# Sử dụng m7i-flex.large: 2 vCPU, 8 GB RAM — Free Tier eligible, đủ mạnh cho LightGBM
resource "aws_instance" "ml_node" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "m7i-flex.large"
  subnet_id              = aws_subnet.private[0].id
  vpc_security_group_ids = [aws_security_group.ml_sg.id]
  key_name               = aws_key_pair.lab_key.key_name

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    hf_token = var.hf_token
    model_id = var.model_id
  })

  tags = { Name = "ML-Inference-Node" }
}

# 6. Load Balancer
resource "aws_lb" "ml_alb" {
  name               = "ml-inference-alb-${random_id.id.hex}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [for subnet in aws_subnet.public : subnet.id]
}

resource "aws_lb_target_group" "ml_tg" {
  name     = "ml-inference-tg-${random_id.id.hex}"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.ml_vpc.id

  health_check {
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

resource "aws_lb_listener" "ml_listener" {
  load_balancer_arn = aws_lb.ml_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ml_tg.arn
  }
}

resource "aws_lb_target_group_attachment" "ml_tg_attach" {
  target_group_arn = aws_lb_target_group.ml_tg.arn
  target_id        = aws_instance.ml_node.id
  port             = 8000
}