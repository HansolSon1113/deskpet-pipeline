# 보드 하드웨어 통합

`DeskPetHardwareIntegration` target은 형제 저장소 `../deskpet-hardware/swift`를 사용합니다. 기존 `DeskPetPipeline` 코어는 하드웨어 모듈을 import하지 않습니다. 두 소스를 나란히 두고 Swift 6.0.3으로 빌드합니다.

```sh
git clone https://github.com/hansung-capstone-team-8-new-line/deskpet-hardware.git
git clone https://github.com/hansung-capstone-team-8-new-line/deskpet-pipeline.git
swift test --package-path deskpet-pipeline
```

`HardwareBridge`는 `SpeakerAdapter`, `DisplayAdapter`, `ManagedMicrophoneSource`를 구현합니다. 한 `HardwareClient`를 공유하며 `start(pipeline:)`가 터치·재생·입력 무효화 이벤트를 단독 소비합니다. 다른 코드가 같은 `client.events`를 소비하면 안 됩니다.

마이크는 16kHz mono s16le 20ms입니다. `AudioIngressConfiguration.sampleRate=16000`, `maxFrameBytes=640`으로 설정합니다. 실제 VAD를 사용할 때는 `ResettableVoiceActivityDetector`를 구현해 `HardwareBridge`와 `AudioIngress`에 같은 인스턴스를 주입하세요. `BoardVAD/BoardSTT`는 보드 연결 시험 전용 합성 fixture입니다.

추가 인터페이스:

- `AudioFrame.inputEpoch`: 선택적 입력 세대. 기존 mock은 nil로 호환되며 하드웨어 연결은 실제 epoch를 보존합니다.
- `invalidateInput(epoch:)`, `resumeInput(epoch:)`, `acceptsInput(epoch:)`: 음성 입력 admission과 늦은 결과 무효화. 재생은 실제 완료+300ms 뒤 듣기 화면 표시를 거쳐 재개합니다.
- `InteractionEvent.touch(TouchInput)`, `PipelineEvent.touch`: 진단 전달만 수행하고 업무 실행·취소·알림 확인으로 해석하지 않습니다.
- `ActivityDisplayAdapter`: 처리 중·질문 대기 등 화면 상태를 직접 전달합니다. 유실 가능한 진단 이벤트 스트림에 화면 동작을 의존시키지 않습니다.
- `HardwareBridge`의 `motionEnabled`는 기본 false입니다. 실물 보정 완료 뒤 활성화하며 듣기 중에는 움직임을 억제합니다.

스피커는 첫 청크에서 포맷을 정하고 뒤의 변경을 거절합니다. 실제 drain ack만 재생 완료이며 비동기 `stop` 오류는 진단에 기록하고 입력 게이트를 임의로 열지 않습니다. 화면 오류와 귀 오류는 업무 실행 상태를 변경하지 않습니다.

```sh
# 먼저 하드웨어 데몬을 board-only 설정으로 실행
swift run deskpet-board --seconds 10
# 물리 장치용 데몬에 연결하되 모델은 계속 fixture
swift run deskpet-board --real
```

전체 설치·배포·30분 시험은 형제 저장소의 `docs/board-test.md`를 따릅니다. 테스트 중 `DESKPET_TEST_SOCKET`을 지정하면 BridgeTests가 실제 소켓으로 실행됩니다. 지정하지 않으면 해당 실소켓 테스트는 SKIP이며 단위 테스트만 통과한 것을 통합 성공으로 간주하지 않습니다.
