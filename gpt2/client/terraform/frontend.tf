# Create the bucket
resource "aws_s3_bucket" "frontend" {
  bucket         = "model-frontend-bucket"
  force_destroy  = true
}

# Disable block public access settings
resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# Enable static website hosting
resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  index_document {
    suffix = "index.html"
  }
}

# Create bucket policy
resource "aws_s3_bucket_policy" "frontend" {
  depends_on = [aws_s3_bucket_public_access_block.frontend]
  bucket = aws_s3_bucket.frontend.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.frontend.arn}/*"
      },
    ]
  })
}



# CloudFront distribution
resource "aws_cloudfront_distribution" "frontend_distribution" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Frontend Distribution"
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_id   = "S3-frontend-bucket"

    s3_origin_config {
      origin_access_identity = ""
    }
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-frontend-bucket"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  price_class = "PriceClass_100"
}

output "frontend_distribution_domain_name" {
  value = aws_cloudfront_distribution.frontend_distribution.domain_name
}


# Generate the index.html from template
resource "local_file" "index_html" {
  content = templatefile("../frontend/index.html.tpl", {
    server_url = "http://${data.terraform_remote_state.client.outputs.instance_private_ip}:8000/generate"
    username = "UStAilaN"
    password = "pK9#mJ4$xL2@"
  })
  filename = "../frontend/index.html"
}

# Upload to S3
resource "aws_s3_object" "frontend_object" {
  bucket = aws_s3_bucket.frontend.id
  key    = "index.html"
  content = local_file.index_html.content
  content_type = "text/html"
}

# Add this to your frontend.tf
data "terraform_remote_state" "client" {
  backend = "local"  # or whatever backend you're using

  config = {
    path = "terraform.tfstate"  # adjust path to your client state file
  }
}