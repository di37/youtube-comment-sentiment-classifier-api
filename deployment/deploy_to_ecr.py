#!/usr/bin/env python3
"""
AWS ECR Deployment Script for YouTube Sentiment Insights
This script uses boto3 and docker-py to build and push Docker images to AWS ECR
No AWS CLI required!

Prerequisites:
--------------
1. Install dependencies: boto3 and docker (already in requirements.txt)
   pip install boto3 docker

2. Configure AWS credentials (one of these methods):
   
   Method 1 - Environment Variables:
   export AWS_ACCESS_KEY_ID='your_access_key'
   export AWS_SECRET_ACCESS_KEY='your_secret_key'
   export AWS_DEFAULT_REGION='me-central-1'
   
   Method 2 - AWS Credentials File (~/.aws/credentials):
   [default]
   aws_access_key_id = your_access_key
   aws_secret_access_key = your_secret_key
   region = me-central-1
   
   Method 3 - IAM Role (if running on EC2/ECS/Lambda)
   No configuration needed - credentials are automatically provided

3. Ensure Docker is installed and running

Configuration:
--------------
Set BUILD_PLATFORM to match your deployment environment:
- None: Native platform (e.g., ARM64 on Apple Silicon)
- "linux/amd64": Standard EC2 instances (Intel/AMD)
- "linux/arm64": AWS Graviton instances

The script automatically detects cross-platform builds and uses Docker buildx with emulation.
For example, building for linux/amd64 on Apple Silicon will automatically use buildx.

Usage:
------
# Option 1: Run from deployment directory (recommended)
cd deployment
python deploy_to_ecr.py

# Option 2: Run from project root
python deployment/deploy_to_ecr.py

Smart Features:
- Automatically detects when cross-platform build is needed
- Uses Docker buildx for cross-platform emulation
- Falls back to standard docker build for native builds
- Works from both deployment directory and project root
- No manual configuration needed for most use cases
"""

import boto3
import docker
import base64
import sys
import platform
import subprocess
import os
from botocore.exceptions import ClientError, NoCredentialsError

# Configuration
AWS_REGION = "me-central-1"
AWS_ACCOUNT_ID = "384887233198"
ECR_REPOSITORY = "isham/simple-ml-app"
IMAGE_TAG = "latest"
LOCAL_IMAGE_NAME = "isham/simple-ml-app"

# Build platform - set to None to use native platform, or specify "linux/amd64" for cloud deployment
# Options: None (native), "linux/amd64" (Intel/AMD for standard EC2), "linux/arm64" (ARM for Graviton)
BUILD_PLATFORM = "linux/amd64"  # Set for standard EC2 instances

# Construct ECR registry URL
ECR_REGISTRY = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
ECR_IMAGE = f"{ECR_REGISTRY}/{ECR_REPOSITORY}:{IMAGE_TAG}"

def print_header():
    """Print script header"""
    print("=" * 50)
    print("AWS ECR Deployment Script (Boto3)")
    print("=" * 50)
    print(f"Region: {AWS_REGION}")
    print(f"Registry: {ECR_REGISTRY}")
    print(f"Repository: {ECR_REPOSITORY}")
    print(f"Image Tag: {IMAGE_TAG}")
    print(f"Build Platform: {BUILD_PLATFORM if BUILD_PLATFORM else 'Native (auto-detect)'}")
    print("=" * 50)
    print()

def get_ecr_credentials():
    """
    Step 1: Get ECR authentication credentials using boto3
    Returns the username and password for Docker login
    """
    print("Step 1/4: Getting ECR authentication credentials...")
    
    try:
        # Create ECR client
        ecr_client = boto3.client('ecr', region_name=AWS_REGION)
        
        # Get authorization token
        response = ecr_client.get_authorization_token(
            registryIds=[AWS_ACCOUNT_ID]
        )
        
        # Extract and decode the token
        auth_data = response['authorizationData'][0]
        token = auth_data['authorizationToken']
        
        # Decode base64 token (format is "AWS:password")
        decoded_token = base64.b64decode(token).decode('utf-8')
        username, password = decoded_token.split(':')
        
        registry = auth_data['proxyEndpoint']
        
        print(f"✓ Successfully retrieved ECR credentials")
        print(f"  Registry: {registry}")
        print()
        
        return username, password, registry
        
    except NoCredentialsError:
        print("✗ Error: AWS credentials not found!")
        print("  Please configure AWS credentials using one of these methods:")
        print("  1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("  2. AWS credentials file (~/.aws/credentials)")
        print("  3. IAM role (if running on EC2)")
        sys.exit(1)
        
    except ClientError as e:
        print(f"✗ Error getting ECR credentials: {e}")
        sys.exit(1)

def docker_login(username, password, registry):
    """
    Authenticate Docker client to ECR
    """
    print("Step 1b/4: Authenticating Docker client to ECR...")
    
    try:
        # Initialize Docker client
        docker_client = docker.from_env()
        
        # Login to ECR
        login_response = docker_client.login(
            username=username,
            password=password,
            registry=registry,
            reauth=True
        )
        
        print(f"✓ Docker authentication successful")
        print()
        
        return docker_client
        
    except docker.errors.APIError as e:
        print(f"✗ Error authenticating Docker: {e}")
        sys.exit(1)

def is_cross_platform_build():
    """Check if we're doing a cross-platform build"""
    if not BUILD_PLATFORM:
        return False
    
    current_arch = platform.machine().lower()
    target_arch = BUILD_PLATFORM.split('/')[-1] if BUILD_PLATFORM else None
    
    # Map architectures
    arch_map = {
        'arm64': 'arm64',
        'aarch64': 'arm64',
        'x86_64': 'amd64',
        'amd64': 'amd64',
    }
    
    current = arch_map.get(current_arch, current_arch)
    return target_arch and current != target_arch

def build_docker_image_buildx():
    """
    Build Docker image using buildx (for cross-platform builds)
    """
    print("Step 2/4: Building Docker image with buildx (cross-platform)...")
    print(f"Building: {LOCAL_IMAGE_NAME}:{IMAGE_TAG}")
    print(f"Platform: {BUILD_PLATFORM}")
    print("This may take a few minutes (using emulation)...")
    print()
    
    try:
        # Determine correct paths based on current directory
        # Check if we're in the deployment directory or project root
        current_dir = os.getcwd()
        
        if os.path.basename(current_dir) == 'deployment':
            # Running from deployment directory
            context_path = ".."  # Parent directory is project root
            dockerfile_path = "Dockerfile"  # Dockerfile is in current directory
        else:
            # Running from project root
            context_path = "."
            dockerfile_path = "deployment/Dockerfile"
        
        # Use docker buildx for cross-platform builds
        cmd = [
            "docker", "buildx", "build",
            "--platform", BUILD_PLATFORM,
            "--tag", f"{LOCAL_IMAGE_NAME}:{IMAGE_TAG}",
            "--file", dockerfile_path,
            "--load",  # Load the image into docker images
            context_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Context: {os.path.abspath(context_path)}")
        print(f"Dockerfile: {os.path.abspath(dockerfile_path)}")
        print()
        
        # Run the build command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n✗ Build failed with exit code {process.returncode}")
            sys.exit(1)
        
        print()
        print(f"✓ Build successful")
        print()
        
    except FileNotFoundError:
        print("✗ Error: docker buildx not found!")
        print("  Please ensure Docker Desktop is installed and buildx is enabled.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error building Docker image: {e}")
        sys.exit(1)

def build_docker_image(docker_client):
    """
    Step 2: Build Docker image
    """
    print("Step 2/4: Building Docker image...")
    print(f"Building: {LOCAL_IMAGE_NAME}:{IMAGE_TAG}")
    if BUILD_PLATFORM:
        print(f"Platform: {BUILD_PLATFORM}")
    else:
        print("Platform: Native (auto-detect)")
    print("This may take a few minutes...")
    print()
    
    try:
        # Determine correct paths based on current directory
        current_dir = os.getcwd()
        
        if os.path.basename(current_dir) == 'deployment':
            # Running from deployment directory
            context_path = ".."  # Parent directory is project root
            dockerfile_path = "Dockerfile"  # Dockerfile is in current directory
        else:
            # Running from project root
            context_path = "."
            dockerfile_path = "deployment/Dockerfile"
        
        # Build the image
        build_kwargs = {
            "path": context_path,
            "dockerfile": dockerfile_path,
            "tag": f"{LOCAL_IMAGE_NAME}:{IMAGE_TAG}",
            "rm": True,  # Remove intermediate containers
            "pull": True,  # Pull base images to ensure compatibility
        }
        
        # Only add platform if specified
        if BUILD_PLATFORM:
            build_kwargs["platform"] = BUILD_PLATFORM
        
        print(f"Context: {os.path.abspath(context_path)}")
        print(f"Dockerfile: {os.path.abspath(dockerfile_path)}")
        print()
        
        image, build_logs = docker_client.images.build(**build_kwargs)
        
        # Print build logs
        for log in build_logs:
            if 'stream' in log:
                print(log['stream'], end='')
            elif 'error' in log:
                print(f"Error: {log['error']}")
                sys.exit(1)
        
        print()
        print(f"✓ Build successful")
        print(f"  Image ID: {image.short_id}")
        print(f"  Size: {image.attrs['Size'] / (1024*1024):.2f} MB")
        print()
        
        return image
        
    except docker.errors.BuildError as e:
        print(f"✗ Error building Docker image: {e}")
        sys.exit(1)
    except docker.errors.APIError as e:
        print(f"✗ Docker API error: {e}")
        sys.exit(1)

def tag_image(image):
    """
    Step 3: Tag the image for ECR
    """
    print("Step 3/4: Tagging image for ECR...")
    print(f"Tagging: {LOCAL_IMAGE_NAME}:{IMAGE_TAG} -> {ECR_IMAGE}")
    
    try:
        # Tag the image
        image.tag(ECR_IMAGE)
        
        print(f"✓ Tag successful")
        print()
        
    except docker.errors.APIError as e:
        print(f"✗ Error tagging image: {e}")
        sys.exit(1)

def push_image(docker_client):
    """
    Step 4: Push image to ECR
    """
    print("Step 4/4: Pushing image to AWS ECR...")
    print(f"Pushing: {ECR_IMAGE}")
    print("This may take several minutes depending on image size and network speed...")
    print()
    
    try:
        # Push the image
        push_logs = docker_client.images.push(
            ECR_IMAGE,
            stream=True,
            decode=True
        )
        
        # Print push progress
        for log in push_logs:
            if 'status' in log:
                status = log['status']
                progress = log.get('progress', '')
                layer_id = log.get('id', '')
                
                # Print concise progress updates
                if layer_id:
                    print(f"  {layer_id}: {status} {progress}")
                else:
                    print(f"  {status}")
                    
            elif 'error' in log:
                print(f"✗ Error: {log['error']}")
                sys.exit(1)
        
        print()
        print(f"✓ Push successful")
        print()
        
    except docker.errors.APIError as e:
        print(f"✗ Error pushing image: {e}")
        sys.exit(1)

def print_footer():
    """Print completion message and next steps"""
    print("=" * 50)
    print("Deployment Complete!")
    print("=" * 50)
    print(f"Image successfully pushed to: {ECR_IMAGE}")
    print()
    print("To pull this image on another machine:")
    print(f"  1. Use the deploy_to_ecr.py script")
    print(f"  2. Or: docker pull {ECR_IMAGE}")
    print()
    print("To run with docker-compose:")
    print("  cd deployment && docker-compose up -d")
    print("=" * 50)

def main():
    """Main execution function"""
    try:
        # Print header
        print_header()
        
        # Step 1: Get ECR credentials and login
        username, password, registry = get_ecr_credentials()
        docker_client = docker_login(username, password, registry)
        
        # Step 2: Build Docker image
        # Use buildx for cross-platform builds, regular build for native
        if is_cross_platform_build():
            print(f"⚠️  Cross-platform build detected: building for {BUILD_PLATFORM} on {platform.machine()}")
            print("   Using Docker buildx for emulation")
            print()
            build_docker_image_buildx()
            # Get the image object after buildx build
            try:
                image = docker_client.images.get(f"{LOCAL_IMAGE_NAME}:{IMAGE_TAG}")
            except docker.errors.ImageNotFound:
                print("✗ Error: Image not found after build. This may be a buildx issue.")
                print("  Trying to continue with tagging...")
                image = None
        else:
            image = build_docker_image(docker_client)
        
        # Step 3: Tag image for ECR
        if image:
            tag_image(image)
        else:
            # Manually tag if image object not available
            print("Step 3/4: Tagging image for ECR...")
            try:
                docker_client.api.tag(
                    f"{LOCAL_IMAGE_NAME}:{IMAGE_TAG}",
                    ECR_IMAGE
                )
                print(f"✓ Tag successful")
                print()
            except Exception as e:
                print(f"✗ Error tagging: {e}")
                sys.exit(1)
        
        # Step 4: Push to ECR
        push_image(docker_client)
        
        # Print footer
        print_footer()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n✗ Deployment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

