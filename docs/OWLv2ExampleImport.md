# OWLv2 예시 이미지 등록 안내

OWLv2 모델은 텍스트 대신 **예시 이미지**를 기준으로 탐지를 수행합니다. 저장된 예시 이미지를 데이터베이스에 등록하려면 `scripts/import_owlv2_examples.py` 스크립트를 사용하세요.

## 1. 사전 준비

- GroundingDINO 프로젝트가 설치된 환경(컨테이너 또는 로컬)에서 작업합니다.
- 이미지 파일은 하나의 폴더에 저장되어 있어야 하며, **파일명 자체가 검색어**가 됩니다.
  - 예: `red_backpack.jpg`, `red_backpack-1.png`
  - `이름-숫자` 형식은 숫자를 제거한 텍스트로 저장됩니다.
- 이미지 형식: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` (또는 MIME 타입이 이미지인 파일)
- 기본 데이터베이스 파일 경로:
  - 로컬 실행: `data/sqlite/owlv2_examples.db`
  - Docker 컨테이너: `/opt/program/GroundingDINO/data/sqlite/owlv2_examples.db`

> 데이터베이스 서버를 따로 “실행”할 필요는 없습니다. SQLite 파일을 직접 열기 때문에 존재하지 않으면 자동으로 생성됩니다.

## 2. 실행 명령

```bash
python scripts/import_owlv2_examples.py <이미지_폴더경로>
```

### 주요 옵션

| 옵션 | 설명 |
| --- | --- |
| `--database-url` | 사용할 SQLite URL. 기본값은 프로젝트 설정(`DATABASE_URL`)을 따릅니다. |
| `--recursive` | 하위 폴더까지 재귀적으로 스캔합니다. |

### 예시

```bash
# 기본 DB에 현재 폴더의 이미지를 등록
python scripts/import_owlv2_examples.py ./data/owl_examples

# 하위 폴더까지 포함
python scripts/import_owlv2_examples.py ./data/owl_examples --recursive

# 임의의 DB 파일을 지정
python scripts/import_owlv2_examples.py ./data/owl_examples \
  --database-url "sqlite:////home/kim/GroundingDINO/data/sqlite/owlv2_examples.db"
```

## 3. 처리 결과

스캔이 끝나면 다음과 같은 요약 메시지가 출력됩니다.

```
[done] 처리 완료. 새로 저장: 5, 건너뜀: 2, 총 파일: 7
```

- **새로 저장**: 새 레코드가 추가된 파일 수
- **건너뜀**: 중복 또는 오류로 건너뛴 파일 수
- **총 파일**: 스캔한 전체 이미지 파일 수

중복 판단 기준은 `(정규화된 검색어, 이미지 데이터 해시)`입니다. 같은 검색어에 동일한 이미지가 이미 등록되어 있으면 건너뜁니다.

## 4. 등록 확인

FastAPI 서버가 실행 중이라면 다음 엔드포인트로 확인할 수 있습니다:

```
GET /owlv2/examples?query=<검색어>
```

프론트엔드에서도 OWLv2 모델을 선택한 뒤 동일한 검색어를 입력하면 데이터베이스에 저장된 예시 이미지로 이미지 기반 탐지가 수행됩니다.

## 5. 참고

- 스크립트는 SQLite를 직접 조작하므로 별도의 “DB 실행”이 필요 없습니다.
- Docker 환경에서는 `docker compose up inference`로 컨테이너를 기동한 뒤, 컨테이너 내부에서 스크립트를 실행하거나 볼륨을 마운트한 호스트에서 실행하세요.
- 삭제가 필요하면 FastAPI의 `DELETE /owlv2/examples/{id}` 엔드포인트 또는 SQLite 도구를 이용해 수동으로 제거할 수 있습니다.
