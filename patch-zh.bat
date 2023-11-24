@echo OFF

SET "GAME_HOME=%ProgramFiles(x86)%\Steam\steamapps\common\Backpack Battles Demo" || goto :error
godotpcktool\godotpcktool.exe --pack "%GAME_HOME%\BackpackBattles.pck" --action add --remove-prefix overrides-zh --file overrides-zh || goto :error
echo(
echo Patch done
echo(
pause
goto :EOF

:error
echo(
echo Failed with error #%errorlevel%.
echo(
pause
