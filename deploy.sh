#!/bin/bash

# Oncology Backend Deployment Script
echo "🚀 Deploying Oncology Backend to Modal..."

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please copy env.example to .env and configure your environment variables."
    echo "cp env.example .env"
    exit 1
fi

# Deploy to Modal
echo "📦 Deploying to Modal..."
modal deploy modal_deploy.py

echo "✅ Backend deployment complete!"
echo "🔗 Your backend will be available at the Modal URL shown above"
echo "📝 Update your frontend .env file with the backend URL" 