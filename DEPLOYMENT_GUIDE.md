# Deployment Guide - Emotion Detection & Learning Support Engine

This guide covers deploying your Streamlit app on AWS, Google Cloud, or Azure.

## Prerequisites

- Docker installed locally
- Cloud provider account (AWS, GCP, or Azure)
- Gemini API key (already available)
- GitHub repository with your code

---

## Option 1: Google Cloud Run (Easiest)

### Step 1: Set Up Google Cloud Project

```bash
# Install Google Cloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable necessary APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Step 2: Create .env for Secrets

```bash
# Create a secrets file (NOT to be committed)
echo "GEMINI_API_KEY=your_key_here" > .env.production
```

### Step 3: Deploy to Cloud Run

```bash
# Build and deploy in one command
gcloud run deploy emotion-learning-assistant \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=$(cat .env.production | grep GEMINI_API_KEY | cut -d= -f2)
```

### Step 4: View Logs

```bash
gcloud run logs read emotion-learning-assistant --limit 50
```

---

## Option 2: AWS with Lambda + API Gateway

### Step 1: Prepare for AWS

```bash
# Install AWS CLI
# Download from: https://aws.amazon.com/cli/

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1)
```

### Step 2: Create Docker Image

```bash
# Build Docker image
docker build -t emotion-learning-app:latest .

# Tag for AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag emotion-learning-app:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/emotion-learning-app:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/emotion-learning-app:latest
```

### Step 3: Deploy with ECS/Fargate

```bash
# Use AWS Console or AWS CLI:
# Go to ECS → Create Cluster → Create Task Definition → Create Service
# Use the Docker image URL from ECR
```

### Step 4: Or Use AWS App Runner (Simpler)

```bash
# In AWS Console:
# 1. Go to App Runner
# 2. Create Service
# 3. Select "Container registry" → ECR
# 4. Choose your image
# 5. Configure and deploy
```

---

## Option 3: Azure Container Instances

### Step 1: Setup Azure

```bash
# Install Azure CLI
# Download from: https://learn.microsoft.com/cli/azure/

# Login to Azure
az login

# Create resource group
az group create --name emotion-rg --location eastus
```

### Step 2: Build and Push to Azure Container Registry

```bash
# Create container registry
az acr create --resource-group emotion-rg \
  --name emotionregistry --sku Basic

# Build and push image
az acr build --registry emotionregistry --image emotion-learning-app:latest .
```

### Step 3: Deploy Container Instance

```bash
az container create \
  --resource-group emotion-rg \
  --name emotion-app \
  --image emotionregistry.azurecr.io/emotion-learning-app:latest \
  --cpu 2 \
  --memory 1 \
  --registry-login-server emotionregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --environment-variables GEMINI_API_KEY='your_key_here' \
  --ports 8501 \
  --dns-name-label emotion-learning-app \
  --protocol TCP
```

---

## Environment Variables in Production

Create environment variable secrets in your cloud provider:

| Variable | Value |
|----------|-------|
| `GEMINI_API_KEY` | Your API key |
| `DATABASE_PATH` | `/app/data/learning_assistant.db` |

---

## Testing Locally with Docker

Before deploying, test the Docker image locally:

```bash
# Build image
docker build -t emotion-learning-app:latest .

# Run container
docker run -p 8501:8501 \
  -e GEMINI_API_KEY="your_key_here" \
  emotion-learning-app:latest

# Access at: http://localhost:8501
```

---

## Database Persistence

For SQLite persistence across deployments:

### Google Cloud Run:
- Use Google Cloud Storage
- Or use Cloud Firestore instead of SQLite

### AWS:
- Use RDS (PostgreSQL/MySQL) instead of SQLite
- Or EBS volumes with ECS

### Azure:
- Use Azure Cosmos DB
- Or Azure Database for PostgreSQL

---

## Monitoring & Logs

### Google Cloud Run:
```bash
gcloud run logs read emotion-learning-assistant --limit 100
```

### AWS CloudWatch:
```bash
aws logs tail /ecs/emotion-learning-app --follow
```

### Azure:
```bash
az container logs --resource-group emotion-rg --name emotion-app --follow
```

---

## Troubleshooting

### Port Issues
Ensure the app uses port 8501 and binds to 0.0.0.0:
```python
# In app.py, this is already handled by:
# streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Memory Issues
If the app crashes, increase memory:
- Google Cloud Run: Set memory to 2GB
- AWS Fargate: Increase task memory to 2GB
- Azure: Increase memory to 2GB

### Database Lock Issues
If using SQLite in production:
- Consider switching to PostgreSQL
- Or implement connection pooling
- Or use Cloud Firestore/Cosmos DB

---

## Next Steps

1. **Choose your cloud platform** above
2. **Test locally with Docker first**
3. **Set up environment variables** in cloud console
4. **Deploy using the commands above**
5. **Monitor logs** for errors
6. **Optimize database** if needed

For questions, refer to your cloud provider's documentation.
