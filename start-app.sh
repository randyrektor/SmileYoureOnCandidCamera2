#!/bin/bash

# React Baby Development Environment Starter
# This script starts all development processes

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting React Baby Development Environment...${NC}"

# Change to project directory
cd "$(dirname "$0")"

# Kill any existing processes on ports 8000 and 5173
echo -e "${BLUE}Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 3  # Give more time for processes to fully terminate

# Start backend server in background
echo -e "${BLUE}Starting backend server...${NC}"
cd backend
# Always activate the venv before running uvicorn
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo -e "${RED}❌ Python venv not found!${NC}"
  exit 1
fi

# Start backend with correct module path
PYTHONPATH="$(pwd)/backend" python -m src.main > ../backend.log 2>&1 &
BACKEND_PID=$!

# Verify backend process started
sleep 2
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Backend failed to start!${NC}"
    echo -e "${RED}💡 Check backend.log for errors${NC}"
    exit 1
fi

echo $BACKEND_PID > ../.dev_pids
cd ..

# Test backend connectivity with timeout and retries
echo -e "${BLUE}Testing backend connectivity...${NC}"
sleep 8  # Give backend more time to start and load models

# Retry logic for backend connectivity
MAX_RETRIES=5
RETRY_COUNT=0
BACKEND_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$BACKEND_READY" = false ]; do
    echo -e "${BLUE}Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES: Testing backend...${NC}"
    
    if curl -s --max-time 15 http://localhost:8000/api/available-videos > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is responding!${NC}"
        echo -e "${GREEN}🚀 MediaPipe face detection is ready${NC}"
        BACKEND_READY=true
    else
        echo -e "${BLUE}⏳ Backend not ready yet, waiting...${NC}"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 5
    fi
done

if [ "$BACKEND_READY" = false ]; then
    echo -e "${RED}❌ Backend failed to start after $MAX_RETRIES attempts${NC}"
    echo -e "${RED}💡 Check backend.log for errors${NC}"
    echo -e "${RED}💡 You may need to manually start the backend:${NC}"
    echo -e "${BLUE}   cd backend && source venv/bin/activate && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000${NC}"
    exit 1
fi

# Start frontend server in background
echo -e "${BLUE}Starting frontend server...${NC}"
cd frontend

# Check if node_modules exists, if not run npm install
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    npm install
fi

npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

# Verify frontend process started
sleep 2
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Frontend failed to start!${NC}"
    echo -e "${RED}💡 Check frontend.log for errors${NC}"
    exit 1
fi

echo $FRONTEND_PID >> ../.dev_pids
cd ..

echo -e "${GREEN}✅ Development environment started!${NC}"
echo -e "${BLUE}Backend: http://localhost:8000${NC}"
echo -e "${BLUE}Frontend: http://localhost:5173${NC}"
echo ""

# Open browser automatically
echo -e "${BLUE}Opening browser...${NC}"
sleep 3
open http://localhost:5173

# Keep the terminal open and show server logs
echo -e "${BLUE}Servers are running. Showing logs (Ctrl+C to stop):${NC}"
echo ""

# Function to handle Ctrl+C gracefully
cleanup() {
    echo ""
    echo -e "${GREEN}✅ Log display stopped.${NC}"
    echo -e "${BLUE}Stopping all development servers...${NC}"
    
    # Run the stop script to properly clean up all processes
    ./stop-app.sh
    
    exit 0
}

# Set up signal handler for Ctrl+C
trap cleanup SIGINT

# Tail the log files to show real-time output
tail -f backend.log frontend.log 