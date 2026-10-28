#!/bin/bash
# =====================================
# Fixed Script: run_weaviate.sh
# =====================================

CONTAINER_NAME="weaviate"
IMAGE_NAME="semitechnologies/weaviate:latest"
PORT_HTTP=8080
PORT_GRPC=50051

echo "[INFO] Checking Weaviate status..."

RUNNING=$(docker ps -q -f name="^${CONTAINER_NAME}$")
EXISTS=$(docker ps -aq -f name="^${CONTAINER_NAME}$")

if [ -n "$RUNNING" ]; then
    echo "[OK] Weaviate is already running."
    exit 0
elif [ -n "$EXISTS" ]; then
    echo "[INFO] Starting existing container..."
    docker start $CONTAINER_NAME
    exit 0
else
    echo "[INFO] Creating new Weaviate container with OpenAI support..."
    docker run -d \
      --name $CONTAINER_NAME \
      -p ${PORT_HTTP}:${PORT_HTTP} \
      -p ${PORT_GRPC}:${PORT_GRPC} \
      -v weaviate_data:/var/lib/weaviate \
      -e QUERY_DEFAULTS_LIMIT=25 \
      -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
      -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
      -e DEFAULT_VECTORIZER_MODULE='text2vec-openai' \
      -e ENABLE_MODULES='text2vec-openai' \
      $IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] New Weaviate container launched!"
    else
        echo "[ERROR] Failed to launch Weaviate."
        exit 1
    fi
fi