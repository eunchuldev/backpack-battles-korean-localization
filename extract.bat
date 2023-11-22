@echo OFF

SET "GAME_HOME=%ProgramFiles(x86)%\Steam\steamapps\common\Backpack Battles Demo" || goto :error
godotpcktool.exe --pack "%GAME_HOME%\BackpackBattles.pck" --include-regex-filter en.translation  --action extract --output extracted || goto :error
