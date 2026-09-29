# Wiki 계약 대응표

기준: `$VAULT_DIR/wiki/deskpet-interface-contracts.md` (updated: 2026-09-29),
`deskpet-technology-stack.md`, `small-router-model.md`, `eureka-process.md`, `deskpet-poc-validation.md`.
Wiki의 제안 타입/timeout을 구현 시작점으로 채택했습니다. 사용자 요청에 따라 I-06/I-07의 API 실행은 별도 큐 없이 직접 비동기 호출하며, 경로별 동시 실행 한도로 과부하를 제한합니다. 미확정 hardware enum과 실제 서버 response schema는 확정하지 않았습니다.

| Wiki ID | 코드 경계 | 구현 범위 |
| --- | --- | --- |
| I-01 | `MicrophoneSource`, `AudioFrame` | pull capture 경계·포맷/프레임 크기 검사; 실제 드라이버는 주입 |
| I-02 | `VoiceActivityDetector`, `AudioIngress` | CPU VAD 프레임 즉시 전달, overflow 발화 무효화·오류 보고, 재생 중 입력 지속; 실제 VAD/AEC는 주입 |
| I-03 | STT frame queue, `StreamingSpeechRecognizer`, 호환 `submitAudio` | 프레임 admission 후 즉시 반환·제출/완료 처리 분리·순서 보존; 기본 mock은 합성 UTF-8 fixture |
| I-04 | `submit`, `Transcript`, router mailbox | partial 진단만, final 처리, confidence·빈 입력·deadline gate |
| I-05 | `RouterModel`, `DecisionComposer` | typed heads, intention masking, confidence, 실행 경로 코드 조립 |
| I-06 | Eureka 모듈 제출/완료, `ActionRequest` | async request/reply, operation/request correlation, 필수 슬롯 gate |
| I-07 | Harness 모듈 제출/완료, `HarnessRequest/Result` | async request/reply, mutation approval 기본 false, 진행 안내·deadline |
| I-08 | `HarnessAdapter` | LLM Wiki 검색은 실제 Harness 내부 구현; mock만 제공 |
| I-09 | `HarnessAdapter` | 실제 MCP 도구·권한·승인·tool별 timeout은 실제 Harness 책임; mock read 결과 |
| I-10 | `EurekaAdapter` | 실행/조회/상태 확인 protocol 및 상태를 바꾸는 mock; HTTP adapter 미포함 |
| I-11 | `pollReminders`, reminder history | 5분 loop·24시간 필터·1건 선택·최신 상태 확인·enqueue/전달 이력 분리 |
| I-12 | output mailbox, `OutputRequest` | action status/description/dueAt/metadata 전달 |
| I-13 | `OutputModel`, `generate` | timeout·빈/부적합 문장 fallback; LangChain chain은 외부 adapter |
| I-14 | TTS 모듈, `SpeechSynthesizer`, `PCMChannel` | output ID·reminder key 보존·첫 chunk timeout |
| I-15 | `RemoteRouter` | 로컬 heads 실패 fallback protocol·deadline 전달; 배포/HTTP 구현 미포함 |
| I-16 | `AudioSource`, `SpeakerAdapter` | 독립 재생 모듈·bounded PCM channel, job/output correlation, finish=physical completed |
| I-17 | `DisplayAdapter`, `OutputEvent` | displayed ack·LCD fallback·15초 복귀; 지원되지 않은 motion은 생성하지 않음; GPIO/servo/LED는 장치 adapter |
| I-18 | `InteractionEvent`, `interact` | speech-start/cancel/ack/snooze; touch 좌표·debounce는 드라이버 |
| I-19 | 웹 모듈 입력/완료 큐, `WebConsumer`, events | 음성·LCD와 독립 처리·소비자 오류 격리; 실제 웹 UI/transport 미포함 |
| I-20 | `RouterContextStore.delivered` | 실제 리마인더 전달 문장을 대상 assistant 턴으로 저장 |
| I-21 | `RouterContextStore.commit` | final user·typed route 순차 저장, context stamp 확인·중복 turn 방지 |
| I-22 | `RouterContextStore.prepare`, `RouterInput` | 대상 전환·유한 UTF-8 budget·role 문맥 snapshot; tokenizer는 adapter |
| I-23 | `RouterContextStore.delivered` | 전달 완료·대상/context 확인·output 중복 방지; 일반 잡담/진행 제외 |

## 어댑터 구현 계약

- 네트워크 본문, 모델 출력 텐서/문장, 기기 내부 큐도 크기를 제한합니다. 파이프라인에서 반환 뒤 검사하는 것으로 adapter 내부의 무제한 할당을 막을 수 없습니다.
- Eureka write를 내부에서 재시도하지 않습니다. 응답 유실을 confirmed failure로 바꾸지 않습니다. 읽기 retry가 필요하면 1회 이내로 제한합니다.
- Eureka `stage$$` 완료/보류 및 삭제·기한 변경을 정규화한 최신 reminder item을 반환합니다. 목록 pagination을 생략하고 전체 목록이라고 보고하지 않습니다.
- Harness는 낮은 confidence·모호한 대상·미확정 slot을 추측해 실행하지 않습니다. `mutationApproved == false`이면 MCP 변경 도구를 호출하지 않습니다. `wikiSearchAllowed`와 typed context packet을 활용합니다.
- `SpeakerAdapter.finish`와 `DisplayAdapter.show`는 접수 시점이 아닌 완료 ack를 기다립니다. IDs로 오래된 작업을 차단하고 `stop`은 신속히 동작해야 합니다.
- `AudioSource`는 샘플 포맷을 유지하고 chunk를 bounded size로 반환합니다. 예시는 signed PCM16, 24 kHz mono로 시작하지만 실제 보드 포맷은 별도 확정합니다.
- SDK·C++ 추론이 blocking이면 Swift cooperative executor를 막지 않습니다. 전용 executor/process와 취소 지원을 제공합니다.
- `ContractCodec`를 사용하면 envelope Date는 RFC 3339 UTC로 직렬화됩니다. `dueAtEpochMs`는 `DateCodec.epochMilliseconds`로 만들며 기한 안내는 원 시간대를 보존합니다.
- 공통 envelope는 wiki 표의 `parentId`를 사용합니다. wiki 흐름 예시의 `causationId`는 같은 인과 관계를 표현하는 다른 제안 이름이므로 실제 서비스 연동 때 하나로 합의해야 합니다.
- 출력의 `motion`은 기본 nil입니다. 실제 지원 enum·servo 범위·표정 map을 장치 담당자와 합의하기 전 임의의 장치 동작을 생성하지 않습니다.

## 보드 통합 검증

1. 실제 한국어 STT 입력에서 partial/final·낮은 confidence·재생 중 입력과 실제 AEC를 확인합니다.
2. 대상 세 번 연속 수정·동명 대상·대상 전환·질문 응답/만료를 검증합니다.
3. 테스트 계정에서 write 성공/403/404/409/응답 유실을 재현하고, 쓰기 호출 횟수와 reconciliation 결과를 확인합니다.
4. 음성·LCD·웹 각각 실패시 업무 결과 보존과 완료 event 의미를 확인합니다.
5. 5분 polling과 24시간 경계·기한 변경·완료·중복·snooze를 검증합니다.
6. 보드/OS/런타임을 함께 기록하고 p50/p95·peak RSS/GPU memory·온도·30분 crash/OOM 유무를 측정합니다.

모듈별 제출/완료 처리 서브모듈·단일 GPU 전담 스레드와 순서 제약은 `docs/concurrency-design.md`를 따른다. 현재의 async API 직접 호출 정책은 유지한다.

GPU 연산은 각 모델 모듈이 구성하고 `GPUExecutor`에 제출한다. GPU 실행/응답은 mock이며 모델 소유권은 모듈에 둔다. TensorRT는 확정하지 않았고 STT/TTS 배치는 미정이다.
