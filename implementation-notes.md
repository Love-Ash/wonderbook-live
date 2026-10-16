# Implementation Notes — SafeShare Lens (공유 전 사진 민감정보 맥락 감지·가림 점검 도구)

## 10줄 이내 구현 계획 및 검사 명령
1. **목표**: 사진·스크린샷을 공유(LinkedIn 게시, 중고거래/택배 인증, 버그 리포트 공유)하기 전, 공유 목적(맥락)에 따라 가려야 할 민감정보(명찰 연락처, 배지 QR, 모니터 API 키, 송장 주소/전화번호 등)를 감지·설명하고 사용자가 수정·가린 뒤 사후 재검증하는 도구 구현 (`기획검토/일상용_사진서비스_후보비교.md` 1번 안).
2. **`app.py` (고정 이미지 처리 및 검증 루프)**:
   - `Pillow` 기반 합성 샘플 이미지 3종(컨퍼런스 명찰·노트북 화면, 택배 송장·계좌 메모, 대시보드 에러 캡처) 생성기 및 이미지 SHA-256 해시/해상도 고정.
   - `gemini-4-argon` (`thinking_level="medium"`, 금지 레거시 파라미터 제외, 최대 2회 오류 로그 재시도) 기반 **1차 맥락 감지(`/api/analyze`)** 및 **2차 가림 후 잔여 노출 재점검(`/api/verify-redaction`)**.
   - 고정 서버 마스킹 함수(`apply_redaction_boxes`: `solid` / `pixelate` / `blur`, EXIF 메타데이터 완전 제거) 및 **결정론적 픽셀 변경 검증**(마스킹 영역의 픽셀 변화율·불투명도·경계 검증, 실패 시 `execution_blocked`).
   - 상태 구분: `detection_needs_review` (모델 감지 제안 단계), `redaction_verified` (사용자 확인 + 픽셀 마스킹 검증 + 2차 재점검 완료), `execution_blocked` (손상된 이미지, 빈 박스, 범위 밖 좌표 등).
   - 실행별 원본 해시, 공유 맥락, 감지/수정 박스, 픽셀 검증 수치, 재점검 결과를 로컬 및 Google Cloud Storage(`gs://$ARTIFACT_BUCKET/runs/<run_id>/record.json` + 가림 이미지)에 저장 후 재읽기 대조.
3. **`static/index.html` (영어 단일 웹 화면)**:
   - 공유 목적 선택, 샘플 사진 로드 및 내 사진 업로드, 캔버스 위 민감정보 테두리(Box) 오버레이 및 판단 이유 카드 표시.
   - 사용자 체크박스 토글(남길 정보 해제 / 가릴 정보 선택) + 마우스 드래그로 빠진 영역 직접 추가 + 전후 나란히 보기 & Before/After 슬라이더 비교 + 사후 재점검 리포트 및 최종 클린 PNG 다운로드.
4. **배포 및 운영 준비**: `Dockerfile` (`Pillow`, `google-genai`, `google-cloud-storage` 포함), `requirements.txt`, `RUN_GOOGLE.ps1`, `run_google.sh`, `deploy.sh` 작성.
5. **검사 명령**:
   - `python -m unittest discover -s tests -p "test_*.py" -v`
   - `python -m unittest discover -s "C:\Users\whywh\OneDrive\Google_DeepMind_해커톤_2026-10-16\실행패키지\qa" -p "test_*.py" -v`
