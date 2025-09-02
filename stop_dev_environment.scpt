-- AppleScript to stop React Baby development environment
-- This script kills all development processes

-- Kill all development processes
do shell script "pkill -f 'vite'"
do shell script "pkill -f 'uvicorn'"
do shell script "pkill -f 'npm run dev'"
do shell script "pkill -f 'esbuild'"

-- Close Terminal windows with specific titles
tell application "Terminal"
	set windowList to every window
	repeat with currentWindow in windowList
		set tabList to every tab of currentWindow
		repeat with currentTab in tabList
			if custom title of currentTab contains "React Baby" then
				close currentWindow
				exit repeat
			end if
		end repeat
	end repeat
end tell

-- Display notification
display notification "React Baby development environment stopped!" with title "Dev Environment" subtitle "All servers terminated" 