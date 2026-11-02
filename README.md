# DeskPet Swift Pipeline

모델·서버·장치 구현은 교체 가능한 protocol로 분리했고 기본 실행은 합성 fixture와 mock만 사용합니다.
구조도보다 wiki의 역할·컨텍스트·완료 이벤트 계약을 우선했습니다. 패키지는 외부 의존성이 없습니다.

```sh
swift test
swift run deskpet-demo
swift run -c release deskpet-demo --module-demo
swift build -c release
/usr/bin/time -l .build/release/deskpet-demo --benchmark
```

Xcode에서 `Package.swift`를 열어 사용할 수 있습니다. macOS에서 빌드·검증했고, Linux/Jetson/Pi 빌드는 아직 검증하지 않았습니다.

## 한 번에 실행

모델과 기존 Harness `.env`가 준비된 상태에서 실행합니다. macOS에서는 `run.command`를 더블클릭해도 됩니다.

```sh
./run.command
```

마이크 대신 키보드로 입력하려면 `./run.command --keyboard`를 실행하거나 `run-keyboard.command`를 더블클릭합니다. 문장을 입력한 뒤 Enter로 전송하고 `/quit` 또는 `Ctrl-C`로 종료합니다. STT·VAD 모델은 로드하지 않으며, 기존 리마인드·라우터·Harness·음성 출력은 사용합니다. 키보드에서도 승인·업무 변경을 전달하며 확정 입력 신뢰도는 1입니다. 읽기만 실행하려면 `--read-only`를 명시합니다.

`./run.command --keyboard --direct-llm`은 라우터를 로드하지 않고 기존 Harness의 Gemini로 바로 전달합니다. 모델별 시간과 프로세스 메모리는 실행마다 `.build/performance/` 아래 CSV로 기록합니다. 옵션과 측정 범위는 [모델 성능 기록](docs/model-performance.md)을 참고하세요.

하드웨어 모의 화면과 Harness를 빌드·실행한 뒤 Mac 마이크·스피커를 사용하는 실제 파이프라인을 시작합니다. `Ctrl-C`로 함께 종료합니다. 처음 실행할 때 `Models/native.local.json`을 예제에서 만들며 기존 설정은 유지합니다. 실행 절차와 옵션은 [네이티브 모델 연결](docs/native-models.md)을 참고하세요.

## 모듈 구조

각 모듈은 모델/API 클라이언트 등의 역할 자원을 소유하고, 작업 제출과 완료 처리를 분리합니다.
`PipelineModule`에는 제출 루프 하나와 완료 처리 루프 하나가 있습니다. 모든 모듈이 독립적으로 진행하며
같은 요청 안에서 결과가 필요한 작업만 기다립니다.

```mermaid
flowchart LR
  Input[모듈 입력 큐] --> Submit[작업 제출 서브모듈]
  Submit --> Runtime[GPU 실행부 / API / CPU 실행부]
  Runtime --> CompleteQueue[완료 큐]
  CompleteQueue --> Process[완료 처리 서브모듈]
  Process --> Next[후속 모듈 입력 큐]
```

파이프라인에는 STT, 라우터, Eureka, Harness, 출력 모델, TTS, 재생, 화면, 선택적 웹 모듈이 있습니다.
Eureka/Harness는 입력 대기 큐 없이 직접 제출하고, 응답만 각 모듈의 완료 큐로 받습니다.
모델·API 호출은 자원 제한 내에서 진행 중 요청으로 추적합니다. 이는 모듈 실행 루프의 복제가 아닙니다.
장치와 오디오 프레임은 순서를 보존하고, 라우터/출력/API의 독립 요청은 준비된 결과부터 처리합니다.

GPU는 하나의 전담 OS 스레드가 연산 제출·완료 확인·결과 반환을 담당합니다.
모델은 각 모듈에 남으며 실제 GPU 호출은 `MockGPUCall`입니다. TensorRT/CUDA 의존성이나 자체 GPU 병렬
스케줄러는 없습니다. 실제 런타임이 비동기 연산과 완료 handle을 제공하는 경계만 정의했습니다.
[모듈 및 GPU 설계](docs/concurrency-design.md)에 책임과 연결 방식을 정리했습니다.

[클래스 다이어그램](docs/class-diagrams.md)은 전체 소유권, 공통 모듈, 담당자별 통합 protocol과
GPU 실행 경계를 현재 코드 기준으로 표현합니다. 모듈 통합 시 이 문서의 역할별 타입 표와 공개 인터페이스를 참고합니다.

## 사용 예

```swift
let adapters = Adapters(
    stt: MySpeechRecognizer(), router: MyRouter(), eureka: MyEureka(),
    harness: MyHarness(), output: MyOutputModel(), tts: MyTTS(),
    speaker: MySpeaker(), display: MyDisplay()
)
let pipeline = try DeskPetPipeline(adapters: adapters)
await pipeline.start()
let requestId = try await pipeline.submit(Envelope(
    conversationId: "deskpet", kind: "stt.final",
    payload: Transcript(text: "보고서 제출 기한을 금요일로 미뤄줘", sttConfidence: 0.94)
))
// 결과 이벤트를 소비하거나 requestId로 action/delivery를 각각 조회합니다.
let record = await pipeline.record(requestId: requestId)
await pipeline.shutdown()
```

`events`는 용량 제한이 있는 단일 소비자 진단 스트림입니다. consumer가 느리면 이벤트가 유실될 수 있으며 `droppedEventCount()`로 확인합니다. 외부 업무 결과 조회에는 `record(requestId:)`를 사용합니다. 실제 웹 연동은 `WebConsumer` adapter로 주입할 수 있습니다.

## 메모리와 지연 정책

- 모듈 입력 큐 기본 16건, STT 프레임 큐 64건, segment 입력 동시 admission/playback 대기 2건, PCM channel 4 chunk입니다. 라우터/출력의 진행 중 호출은 각 16건, Eureka/Harness는 각 8건, STT/TTS/장치는 각 1건입니다. 이 값은 큐·진행 중 작업의 상한이며 워커 개수가 아닙니다.
- TTS에서 재생 큐로 전달할 때는 빈자리를 기다리는 역압력을 적용합니다. 재생 큐 포화만으로 음성을 폐기하지 않으며 대기 생산자는 하나로 제한합니다.
- 각 진행 중 작업은 완료 큐 슬롯도 예약합니다. 완료 처리까지 슬롯을 유지해 결과 적체 시 무제한 task 생성을 막습니다. API 포화는 호출 전 `API_BUSY`, GPU admission 포화는 `GPU_BUSY`, 입력 큐 포화는 `QUEUE_FULL`입니다.
- 전체 요청 64건, journal/진단 이벤트 256건, 대화 8세션·세션당 16턴/UTF-8 8 KiB, 입력 4 KiB·출력 8 KiB로 제한합니다. native adapter 내부 모델 weights/workspace/KV cache 상한은 해당 모델 모듈 책임입니다.
- `AudioIngress`는 CPU VAD의 speech 프레임을 즉시 STT 큐에 전달합니다. 발화 종료와 추론 완료를 기다려 PCM을 모으지 않으며, 출력 재생으로 입력을 차단하지 않습니다. partial은 진단에만 전달하고 final만 라우팅합니다. STT/TTS의 CPU/GPU 배치는 미정입니다.
- 프레임 큐 또는 발화 크기 상한 초과 시 오류를 보고하고 해당 발화를 무효화해 잘린 명령이 실행되지 않게 합니다. 유한 버퍼로 무제한 입력을 보존할 수 없으므로 실제 캡처 backlog·STT 처리율은 Jetson에서 측정해야 합니다. AEC는 실제 장치 adapter의 통합 대상입니다.
- 라우터 2초, STT/출력 5초, Eureka 8초, Harness 30초, TTS 첫 chunk 3초, 재생 30초·요청 60초는 임시값입니다. `PipelineConfiguration`으로 조정합니다.
- `snapshot()`은 입력 큐와 모듈별 제출·완료 처리·진행 중 호출·완료 큐를 제공합니다. `measurements()`는 최근 256개 표본의 p50/p95를 제공합니다. 실제 모델/GPU 성능 측정을 대신하지 않습니다.

## 실패·취소 정책

`ActionResult`와 `DeliveryResult`를 따로 보존합니다. 성공한 API 요청 뒤 음성이 실패해도 변경을 반복하지 않습니다. 음성 실패 시 동일 `outputId`의 LCD 텍스트로 대체하며, 실제 재생 완료 또는 텍스트 표시 완료만 assistant 턴으로 저장합니다. 표정 표시 ack와 TTS 생성 완료는 발화 전달 완료가 아닙니다.

외부 변경을 dispatch한 뒤 timeout·취소·응답 유실이 발생하면 `unknown`으로 남깁니다. 명시적인 인증·대상 없음·충돌 오류는 `failed`입니다. 같은 요청을 다시 제출하면 중복으로 거부합니다. 미확인 결과는 journal에서 자동 제거하지 않으며, journal이 가득 차면 신규 요청을 거부합니다. `RequestRecord.operation`을 사용해 `reconcile`로 상태만 확인합니다. Harness의 reconciliation은 별도 adapter 구현이 필요합니다.

모든 async adapter는 취소에 협조해야 합니다. GPU 호출 취소는 요청자를 반환시키지만 이미 제출된
연산의 자원은 실제 완료까지 유지합니다. GPU 전담 스레드는 blocking 추론 함수 대신 비동기 제출과
nonblocking 완료 조회를 받습니다. 파이프라인 종료는 모듈의 진행 중 호출을 취소하고 완료 큐를 처리한 뒤
GPU 자원 회수를 기다립니다. 외부 변경의 unknown/confirmed 결과를 큐 폐기로 잃지 않습니다.
완료를 영원히 알리지 않는 런타임의 강제 중단은 현재 구현하지 않았습니다. 오래된 출력은 적용하지 않으며,
장치 adapter도 ID로 취소된 명령과 늦은 ack를 차단해야 합니다.

결과 journal·중복 방지·리마인더 이력은 현재 프로세스 메모리에 있습니다. 재시작을 넘는 쓰기 중복 방지와 `unknown` 복구에는 영속 저장 및 서버 operation ID가 필요합니다. 확인된 완료 요청의 중복 방지도 journal 보존 기간에 한정됩니다.

## 라우터·출력·리마인더

라우터는 대상 해소 → 같은 대상 대화 snapshot → typed heads → intention masking → confidence → 코드 조립 순서로 동작합니다. 모델이 생성한 JSON을 파싱하지 않습니다. STT confidence와 개별/전체 router confidence는 독립적으로 검사합니다. 낮은 STT 신뢰도·필수 슬롯 누락·동명 대상이면 외부 adapter를 호출하지 않고 질문합니다. 낮은 router 신뢰도는 쓰기 승인이 없는 Harness로 전달합니다.

명시적 대상 전환은 이전 대화와 미해결 슬롯을 비웁니다. 아직 대상이 없는 질문에 대한 답은 동일 request에 연결합니다. 다른 대상으로 전환하거나 질문 대기가 만료되면 새 요청으로 분리합니다. 날짜·우선순위 등은 현재 head 값만 사용하고 예전 슬롯을 무조건 복사하지 않습니다.

출력 모델의 사실 검증은 보수적으로 구현했습니다. 업무 결과·리마인더·질문·진행 안내에서는 검증된 고정 문장과 정확히 같을 때만 모델 문장을 사용합니다. 임의 바꿔쓰기의 사실 정합성을 문자열 비교로 증명하지 않습니다. 자연스러운 표현 확장은 별도 typed factual validator가 준비된 뒤 교체할 수 있습니다. 일반 대화만 자유 문장을 허용하며 출력 크기는 제한합니다.

리마인더는 5분 polling, 24시간 이내 필터, `itemId + dueAt + kind` 중복 방지, 출력 직전 최신 설명·기한·상태 재확인을 적용합니다. 완료·보류·삭제 여부는 Eureka adapter가 `eligible`로 정규화해야 합니다. 설명·기한·시간대는 실제 안내에 포함합니다. 재알림 예약은 로컬 이력만 바꾸고 업무 기한을 바꾸지 않습니다. 처리/질문 대기 중에는 안내하지 않고, 음성 완료 뒤 5초, LCD 대체 안내 종료 뒤 5초 동안 알림을 보류합니다.

질문 대기 15초, LCD 텍스트 복귀 15초, 긴 Harness 작업의 10초 진행 안내를 포함합니다. 이 값도 설정으로 조정합니다. 장치 출력 완료를 사용자 확인이나 업무 완료로 간주하지 않습니다.

## 검증

모듈 간 독립 진행, 제출·완료 처리의 분리, 단일 GPU 스레드의 요청별 응답 반환,
취소 후 자원 수명, VAD 프레임의 실시간 전달과 재생 중 입력을 회귀 테스트로 확인합니다.
[검증 범위](docs/validation.md)에 실행 결과와 미검증 경계를 기록합니다.

## 통합 경계

실제 C++/Swift 모델 연결과 TensorRT 교체 경계, 준비·실행 절차는 [네이티브 모델 연결](docs/native-models.md)에 정리되어 있습니다. 기존 mock 실행과 서버 시연은 아래 절차를 그대로 사용합니다.

### 서버 연결 시연

형제 폴더의 `LLM_test`에서 `pnpm web:fixture`를 실행하고 켜 둡니다. 서버 기본 주소는
`ws://127.0.0.1:8788`입니다. 이 저장소에서는 다음 명령을 실행합니다.

```sh
swift run deskpet-demo --connected-demo
```

Mock 발화 `보고서 제출 상태 알려줘` → Eureka 업무 조회, `DeskPet PR 42 리뷰해줘` → Harness PR 리뷰가
Swift pipeline의 실제 서버 adapter를 통해 처리되고, 서버 결과와 최종 발화가 터미널에 표시됩니다.
두 작업의 성공과 출력 완료를 확인하면 `pipeline 출력 완료 2/2`를 출력합니다. 서버 미가동·인증 실패·
업무 실패를 성공으로 표시하지 않습니다. STT/라우터/출력 모델/TTS/장치와 외부 업무 데이터는 mock/fixture입니다.

`--connected-demo`에서만 fixture 주소·전용 토큰을 기본값으로 사용합니다.
다른 서버는 `HARNESS_WS_URL`·`HARNESS_WS_TOKEN`으로 지정합니다. 두 저장소에서 동시에 서버를
새로 실행하지 않으며, `EADDRINUSE`가 나면 기존 서버가 같은 포트를 사용 중인지 확인합니다.
시연은 하드웨어 서버 없이 실행합니다. `DESKPET_SERVER_TESTS=1 swift test`로 두 adapter의
pipeline 연결 회귀를 확인할 수 있습니다.

[계약 대응표](docs/interface-map.md)에 I-01~I-23 연결 지점을 정리했습니다.
`Adapters.gpuMocks(executor:)`는 라우터·출력 모델의 GPU 호출을 mock 처리합니다. 실제 모델·추론 런타임,
Eureka HTTP 스키마·인증·pagination, Harness/MCP·LLM Wiki 검색, 오디오/장치 드라이버·웹 transport는
외부 adapter 통합 대상입니다. STT/TTS 실행 위치는 Jetson 측정 후 정하며 Linux/Jetson 빌드는 미검증입니다.
