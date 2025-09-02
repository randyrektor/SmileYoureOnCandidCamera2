-- AppleScript to start React Baby development environment
-- This script starts all development processes

tell application "Terminal"
	-- Create new window with custom title
	set newWindow to do script ""
	set custom title of front window to "React Baby Dev Environment"
	
	-- Start backend server
	do script "cd $(dirname \"$0\")/backend && source venv/bin/activate && uvicorn src.main:app --reload --port 8000" in newWindow
	
	-- Create new tab for frontend
	tell application "System Events" to keystroke "t" using command down
	delay 1
	
	-- Start frontend server
	do script "cd $(dirname \"$0\")/frontend && npm run dev" in front window
end tell

-- Display notification
display notification "React Baby development environment started!" with title "Dev Environment" subtitle "Backend: localhost:8000, Frontend: localhost:5173" 