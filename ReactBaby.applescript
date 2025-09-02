tell application "Terminal"
    -- Create a new terminal window with a custom title
    set newWindow to do script ""
    set custom title of newWindow to "React Baby Development Environment"
    
    -- Start the development environment with Ctrl+C handling
    do script "cd /Users/randyrektor/react-baby && ./start-react-baby.sh" in newWindow
    
    -- Activate the terminal window
    activate
end tell 