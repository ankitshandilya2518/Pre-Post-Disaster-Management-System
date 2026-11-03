Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\ankit\Desktop\CSE471"
WshShell.Run "Dashboard.bat", 1, False
