#!/bin/bash

# React Baby Development Environment Stopper
# Simple script to stop all development processes

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping React Baby development servers...${NC}"

# Kill processes by name more aggressively
echo -e "${BLUE}Killing development processes...${NC}"
pkill -f "uvicorn" 2>/dev/null || echo "No Uvicorn processes found"
pkill -f "vite" 2>/dev/null || echo "No Vite processes found"
pkill -f "npm run dev" 2>/dev/null || echo "No npm processes found"

# Kill processes on specific ports
echo -e "${BLUE}Killing processes on development ports...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No processes on port 8000"
lsof -ti:5173 | xargs kill -9 2>/dev/null || echo "No processes on port 5173"
lsof -ti:5174 | xargs kill -9 2>/dev/null || echo "No processes on port 5174"

# Clean up PID file if it exists
if [ -f ".dev_pids" ]; then
    echo -e "${BLUE}Cleaning up PID file...${NC}"
    rm .dev_pids
fi

# Clean up log files
if [ -f "backend.log" ]; then
    rm backend.log
fi
if [ -f "frontend.log" ]; then
    rm frontend.log
fi

echo -e "${GREEN}✅ All servers stopped!${NC}" 