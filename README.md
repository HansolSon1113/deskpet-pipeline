# DeskPet Pipeline

DeskPet의 한국어 음성 입력을 처리하는 Swift 파이프라인입니다. 음성 구간 감지(VAD)와 음성 인식(STT)을 거쳐 발화 의도와 업무 대상을 분류합니다. 일상 대화는 로컬 모델이 처리하고, Eureka 업무 조회·변경이나 GitHub PR 검토는 Harness로 전달합니다. 응답과 마감 알림은 음성 합성(TTS)과 화면을 통해 전달합니다.

Swift 패키지는 `DeskPetPipeline` 라이브러리와 `deskpet-demo` 실행 파일을 제공합니다. 모델, 업무 API, 장치는 어댑터로 연결하며 모듈별 입력과 완료 처리는 비동기로 진행합니다.

## 모의 데모

macOS 13 이상과 Swift 6 이상을 기준으로 합니다. Swift 패키지가 `../deskpet-hardware/swift`를 참조하므로 [DeskPet Hardware](https://github.com/hansung-capstone-team-8-new-line/deskpet-hardware)를 파이프라인과 같은 상위 디렉터리에 `deskpet-hardware`라는 이름으로 배치하세요.

```sh
swift test
swift run deskpet-demo
```

기본 데모는 합성 발화와 모의 모델·서비스로 업무 조회, PR 검토, 대상 확인 흐름을 보여줍니다. 데모에는 모델 가중치와 외부 서비스 인증이 필요하지 않습니다.

## macOS에서 음성·키보드로 실행

Apple Silicon Mac에서 [네이티브 모델 준비 안내](docs/native-models.md)에 따라 모델과 런타임을 준비하고 [DeskPet Harness](https://github.com/hansung-capstone-team-8-new-line/deskpet-harness)의 `.env`에 `HARNESS_WS_TOKEN`, `GITHUB_TEST_REPO`, `GITHUB_TEST_PR`과 사용할 서비스의 인증 정보를 설정합니다. 기본 탐색 경로는 형제 디렉터리 `LLM_test`입니다. 다른 위치를 쓸 경우 `Models/native.local.json`의 `harnessEnvFile`과 `DESKPET_HARNESS_DIR`를 해당 체크아웃에 맞춥니다.

```sh
./run.command --check
./run.command
```

`./run.command --keyboard`를 실행하면 터미널에서 문장을 입력할 수 있습니다. 음성 실행은 Mac의 마이크·스피커와 하드웨어 모의 화면을 사용합니다. Jetson에 배포하려면 [DeskPet Product](https://github.com/hansung-capstone-team-8-new-line/deskpet-product)를 사용하세요.

## 기술 문서

| 문서 | 내용 |
| --- | --- |
| [모듈과 동시성](docs/concurrency-design.md) | 비동기 처리, 큐와 모델 실행 경계 |
| [인터페이스](docs/interface-map.md) | 모듈별 프로토콜과 어댑터 |
| [로컬 모델](Models/README.md) | 가중치 준비와 설정 파일 |
| [성능 기록](docs/model-performance.md) | 모델 지연 시간과 메모리 측정 |
| [검증](docs/validation.md) | 개발 환경과 모듈별 시험 절차 |
