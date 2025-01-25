

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # or whatever version you're using
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}


provider "aws" {
  region = "us-east-1"
}