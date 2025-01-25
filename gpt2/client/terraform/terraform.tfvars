# terraform.tfvars
region         = "us-east-1"
instance_type  = "g4dn.xlarge"
environment    = "dev"
project_name   = "model-client"
volume_size    = 60
ssh_allowed_cidr = "0.0.0.0/0"  # Replace with your IP
ami_id         = "ami-0772e0bfc7cc9c5de"

server_vpc_id = "vpc-089d1e09cb9566be3"
