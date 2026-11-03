# SNOWISE DISCOVER

숙명여자대학교 비교과 프로그램 참여 데이터 분석 및 추천 시뮬레이션 Streamlit 프로토타입입니다.

## 1. GitHub 저장소 구성

```text
SNOWISE_DISCOVER/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

원본 CSV는 저장소에 올리지 않습니다. 앱을 실행한 뒤 사이드바에서 아래 두 파일을 업로드합니다.

- `03_비교과_프로그램.csv`
- `04_비교과_이수자.csv`

## 2. GitHub에 올리기

1. GitHub에서 새 저장소를 만듭니다.
2. 이 폴더 안의 파일 4개(`app.py`, `requirements.txt`, `README.md`, `.gitignore`)를 저장소 최상위(root)에 올립니다.
3. 학생 식별정보가 포함되거나 가명처리된 원본 CSV는 공개 저장소에 업로드하지 않습니다.

## 3. Streamlit Community Cloud 배포

1. https://share.streamlit.io/ 에 접속하고 GitHub 계정으로 로그인합니다.
2. `Create app`을 선택합니다.
3. 방금 만든 GitHub 저장소를 선택합니다.
4. Branch는 `main`(또는 실제 기본 브랜치), Main file path는 `app.py`로 지정합니다.
5. `Deploy`를 누릅니다.
6. 배포가 끝나면 앱 왼쪽 사이드바에서 위의 CSV 두 개를 업로드합니다.

앱 코드 수정 후 GitHub에 commit/push하면 Streamlit Cloud가 저장소 변경 사항을 반영해 앱을 다시 배포합니다.

## 4. 로컬 실행

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## 5. 포함 기능

- 전체 참여 현황 및 참여 집중도
- 학과별 진단 및 A~D 유형화
- 분야별 프로그램 공급·승인 구조
- 학과 유형별 운영 주체 이용 경로
- 2022~2024 이력 → 2025 신규 참여 Hit@K 오프라인 검증
- 학과·학년·관심 분야·참여 이력 기반 추천 시뮬레이터

## 6. 데이터 및 해석상 주의

- 앱은 사용자가 업로드한 과거 CSV를 처리하는 분석·시연용 프로토타입입니다.
- 추천 시뮬레이터는 과거 데이터의 프로그램 그룹과 사용자가 선택한 조건을 이용한 규칙 기반 시연입니다. 실제 학생별 SNOWISE 추천 결과가 아닙니다.
- 모집 시작일·마감일·정원·실시간 모집 상태 및 SNOWISE 노출·클릭 로그는 데이터에 포함되어 있지 않습니다.
- Hit@K는 과거 승인 이력을 이용한 오프라인 평가값으로, 실제 서비스 전환율이 아닙니다.
- 자율 0회 비율은 제공 데이터에서 승인 기록이 관측된 학생-학년도 중 자율 비교과 승인이 0건인 비율입니다. 전체 재학생의 미참여율로 해석하면 안 됩니다.
- 원본 학생 데이터는 가명처리되어 있더라도 민감할 수 있습니다. 공개 GitHub 저장소에 올리지 말고, 실제 데이터를 처리하는 앱의 접근 권한을 제한해야 합니다.
