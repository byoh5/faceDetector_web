# Face + Plate Masking Web MVP

브라우저에서 사진을 로컬로 처리해 얼굴/번호판을 자동 마스킹하는 MVP입니다.

## 핵심 흐름

1. 접속 후 데이터 안내 확인
2. `변환하러가기` 클릭 시 검출 엔진 1회 다운로드
3. 이미지 업로드(드래그/클릭)
4. 자동 스캔(얼굴 + 번호판) + 기본 마스킹
5. 박스 on/off 또는 수동 박스 추가
6. 단일 JPG 다운로드 또는 여러 장 처리 후 썸네일 검토/승인 뒤 JPG ZIP 일괄 다운로드
7. 기본 데이터 절약 설정(품질/크기 최적화)으로 용량 절감

## 기술 구성

- Frontend: React + Vite + TypeScript
- 얼굴 검출: MediaPipe Face Detector (`@mediapipe/tasks-vision`)
- 번호판 검출: OpenCV.js + Haar Cascade (`haarcascade_russian_plate_number.xml`)
- 렌더링: Canvas 2D

## 실행

```bash
npm install
npm run dev
```

빌드/검증:

```bash
npm run build
npm run lint
```

## GitHub Pages 배포

1. 저장소 루트에 이미 추가된 워크플로우 파일을 확인합니다.
   - `.github/workflows/deploy-pages.yml`
2. GitHub 저장소에서 `Settings > Pages`로 이동합니다.
3. `Build and deployment`의 Source를 `GitHub Actions`로 설정합니다.
4. 기본 배포 브랜치(`main` 또는 `master`)에 푸시하면 자동 배포됩니다.
5. 첫 배포 후 `https://<github-id>.github.io/<repo-name>/`에서 접속합니다.
   - 저장소 이름이 `<github-id>.github.io`인 경우 경로 없이 루트(`/`)로 열립니다.

워크플로우는 `actions/configure-pages` 출력값을 이용해 Vite `base`를 자동 설정하므로,
프로젝트 페이지와 유저 페이지 둘 다 별도 수정 없이 배포됩니다.

## 현재 MVP 범위

- 단일 이미지 처리
- 다중 이미지 일괄 처리 + ZIP 다운로드(JPG)
- 자동 검출 박스 목록/체크박스 제어
- 캔버스 위 박스 클릭 토글
- 수동 드래그 박스 추가
- 마스킹 스타일: 모자이크(기본), 블러, 검은 박스
- JPG 다운로드(기본 품질 82%)
- 해상도 축소(기본 긴 변 1920px)
- 데이터 절약 모드 ON/OFF + 품질/크기 조절
- Web Share API 지원 기기에서 JPG 공유
- 최초 진입 시 데이터 사용 안내 + 동의 후 엔진 다운로드
- 모바일 우선 UI(작은 화면 1열, 큰 화면 확장)

## 주의사항

- 번호판 검출은 Haar Cascade 기반이라 각도, 야간, 원거리 이미지에서 누락이 발생할 수 있습니다.
- 누락을 고려해 수동 박스 보정 UX를 포함했습니다.
- 이미지 처리는 브라우저에서 수행됩니다(로컬 처리).
- 엔진 캐시가 유지되는 같은 브라우저 기준으로 추가 다운로드 없이 재사용됩니다.
