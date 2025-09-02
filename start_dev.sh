#!/bin/bash

# React Baby Development Environment Starter (Alternative)
# This script starts all development processes

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting React Baby Development Environment...${NC}"

# Change to project directory
cd "$(dirname "$0")"

# Start backend server
echo -e "${BLUE}Starting backend server...${NC}"
cd backend
source venv/bin/activate
uvicorn src.main:app --reload --port 8000 &
BACKEND_PID=$!
echo $BACKEND_PID > ../.dev_pids
cd ..

# Start frontend server
echo -e "${BLUE}Starting frontend server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
echo $FRONTEND_PID >> ../.dev_pids
cd ..

echo -e "${GREEN}✅ Development environment started!${NC}"
echo -e "${BLUE}Backend: http://localhost:8000${NC}"
echo -e "${BLUE}Frontend: http://localhost:5173${NC}" 