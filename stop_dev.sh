#!/bin/bash

# React Baby Development Environment Stopper
# This script stops all development processes

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping React Baby Development Environment...${NC}"

# Kill processes by name
echo -e "${BLUE}Killing development processes...${NC}"
pkill -f "vite" 2>/dev/null || echo "No Vite processes found"
pkill -f "uvicorn" 2>/dev/null || echo "No Uvicorn processes found"
pkill -f "npm run dev" 2>/dev/null || echo "No npm processes found"
pkill -f "esbuild" 2>/dev/null || echo "No esbuild processes found"

# Kill processes by PID if .dev_pids file exists
if [ -f ".dev_pids" ]; then
    echo -e "${BLUE}Killing processes by PID...${NC}"
    while IFS= read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done < .dev_pids
    rm .dev_pids
fi

# Kill processes on specific ports
echo -e "${BLUE}Killing processes on development ports...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No processes on port 8000"
lsof -ti:5173 | xargs kill -9 2>/dev/null || echo "No processes on port 5173"

echo -e "${GREEN}✅ Development environment stopped!${NC}" 