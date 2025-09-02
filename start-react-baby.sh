#!/bin/bash

# React Baby Development Environment with Ctrl+C handling
# This script starts all development processes and handles graceful shutdown

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to handle cleanup on Ctrl+C
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    
    # Kill processes by name
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "node.*vite" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    
    # Kill processes on specific ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    lsof -ti:5174 | xargs kill -9 2>/dev/null || true
    
    # Clean up files
    rm -f .dev_pids backend.log frontend.log 2>/dev/null || true
    
    echo -e "${GREEN}✅ All services stopped!${NC}"
    exit 0
}

# Set up signal handler for Ctrl+C
trap cleanup SIGINT

echo -e "${BLUE}🚀 Starting React Baby Development Environment...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"
echo ""

# Change to project directory
cd /Users/randyrektor/react-baby

# Kill any existing processes on ports 8000 and 5173
echo -e "${BLUE}🧹 Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 2

# Start backend server
echo -e "${BLUE}🚀 Starting backend server...${NC}"
cd backend

# Always activate the venv before running
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo -e "${RED}❌ Python venv not found!${NC}"
    exit 1
fi

# Start backend with correct module path
PYTHONPATH=/Users/randyrektor/react-baby/backend python -m src.main > ../backend.log 2>&1 &
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
echo -e "${BLUE}🔍 Testing backend connectivity...${NC}"
sleep 8  # Give backend time to start and load models

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
    cleanup
fi

# Start frontend server
echo -e "${BLUE}🎨 Starting frontend server...${NC}"
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
    cleanup
fi

echo $FRONTEND_PID >> ../.dev_pids
cd ..

echo ""
echo -e "${GREEN}🎉 React Baby Development Environment Started!${NC}"
echo -e "${BLUE}📊 Backend: http://localhost:8000${NC}"
echo -e "${BLUE}🎨 Frontend: http://localhost:5173${NC}"
echo -e "${BLUE}📚 API Docs: http://localhost:8000/docs${NC}"
echo ""

# Open browser automatically
echo -e "${BLUE}🌐 Opening browser...${NC}"
sleep 3
open http://localhost:5173

# Show logs and wait for Ctrl+C
echo -e "${BLUE}📋 Showing server logs (Press Ctrl+C to stop all services):${NC}"
echo ""

# Tail the log files to show real-time output
tail -f backend.log frontend.log