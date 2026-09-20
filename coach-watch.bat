@echo off
rem Rocket League coach -- waits for the game, analyses each match you save.
rem
rem Started at login it costs nothing while you are not playing: it checks for
rem RocketLeague.exe every 20 seconds and does nothing else until it appears.
rem
rem The loop is here so the watcher can pick up its own code changes: it exits
rem with code 3 when coach\*.py changes underneath it, and comes straight back
rem on the new code. Any other exit stops for good.
rem
rem Close this window to stop it. Ctrl-C also works.
title Rocket League Coach
cd /d "%~dp0"
echo Rocket League coach -- waiting for the game.
echo.
:run
"%~dp0venv\Scripts\python.exe" -u coach\watch.py --auto
if errorlevel 4 goto done
if errorlevel 3 goto run
:done
echo.
echo Watcher stopped. Press any key to close.
pause >nul
