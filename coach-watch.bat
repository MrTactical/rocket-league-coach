@echo off
rem Rocket League coach -- waits for the game, analyses each match you save.
rem
rem Started at login it costs nothing while you are not playing: it checks for
rem RocketLeague.exe every 20 seconds and does nothing else until it appears.
rem
rem Close this window to stop it. Ctrl-C also works.
title Rocket League Coach
cd /d "%~dp0"
echo Rocket League coach -- waiting for the game.
echo Save a replay in game by HOLDING BACKSPACE at the end of a match.
echo.
"%~dp0venv\Scripts\python.exe" -u coach\watch.py --auto
echo.
echo Watcher stopped. Press any key to close.
pause >nul
