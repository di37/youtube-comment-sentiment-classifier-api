# Deployment Directory

This directory contains all deployment artifacts and scripts for the YouTube Sentiment Classifier API.

## 📁 Contents

### Docker Configuration
- **Dockerfile**: Multi-stage Docker build configuration
- **docker-compose.yaml**: Docker Compose configuration for local development

### CI/CD Scripts
- **deploy_to_ec2.sh**: Main deployment script (used by CI/CD pipeline)
- **setup_ecr.sh**: One-time setup script for AWS ECR repository
- **setup_ec2.sh**: One-time setup script for EC2 instance (Docker + AWS CLI)

## 🚀 Quick Start

### Local Development

Run locally using Docker Compose:

```bash
cd deployment
docker-compose up --build
```

Access the API at: http://localhost:6889

### Production Deployment (CI/CD)

The project includes a fully automated CI/CD pipeline using GitHub Actions. See the [CI/CD Setup Guide](../CICD_SETUP_GUIDE.md) for complete instructions.

**Quick Setup:**

1. **Set up ECR repository:**
   ```bash
   ./setup_ecr.sh
   ```

2. **Prepare EC2 instance:**
   ```bash
   # On EC2 instance
   ./setup_ec2.sh
   ```

3. **Configure GitHub Secrets** (required):
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - EC2_HOST
   - EC2_USER
   - EC2_SSH_KEY
   - MLFLOW_TRACKING_URI

4. **Push to main branch** - pipeline deploys automatically!

## 📚 Documentation

- **[CI/CD Setup Guide](../CICD_SETUP_GUIDE.md)**: Complete setup instructions
- **[GitHub Workflows README](../.github/workflows/README.md)**: Pipeline documentation

## 🔧 Manual Deployment

If you need to deploy without CI/CD:

```bash
# Set environment variables
export AWS_REGION=me-central-1
export ECR_REGISTRY="<account-id>.dkr.ecr.me-central-1.amazonaws.com"
export ECR_REPOSITORY="youtube-sentiment-classifier"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export MLFLOW_TRACKING_URI="http://your-mlflow:5000"

# Run deployment script on EC2
./deploy_to_ec2.sh
```

## 📦 Directory Structure

```
deployment/
├── Dockerfile              # Docker image definition
├── docker-compose.yaml     # Local development configuration
├── deploy_to_ec2.sh       # Production deployment script
├── setup_ecr.sh           # ECR repository setup
├── setup_ec2.sh           # EC2 instance setup
└── README.md              # This file
```
