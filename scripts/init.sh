#!/bin/bash

# Install curl if not already installed (just in case)
apt-get update && apt-get install -y curl

# Start the Ollama app in the background
ollama serve &

# Wait for the Ollama app to start
until curl -s http://localhost:11434/health > /dev/null; do
    echo "Waiting for Ollama to start..."
    sleep 2
done

# Pull the Llama 3.2 model
ollama pull command-r7b

# Keep the container running
tail -f /dev/null
