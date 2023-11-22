# backpack-battles-korean

ChatGPT로 초벌 번역 후, 일부 어색한 번역을 수정하였습니다.

# 패치 적용법

* (게임 설치 경로가 `C:\Program Files (x86)\Steam\steamapps\common\Backpack Battles Demo`가 아닐 경우) patch.bat을 메모장으로 열어 경로 변경
* patch.bat 클릭

# 번역 생성 방법

* `extract.bat` 실행하여 시트 추출
* .env파일에 `OPEN_AI_KEY` 환경변수 설정
* `translate.py translate` 실행하여 번역파일 생성
* `translate.py apply` 실행하여 번역파일 적용
* `patch.bat` 실행
