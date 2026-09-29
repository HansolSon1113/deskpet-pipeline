# 모듈 파이프라인 검증

검증일: 2026-09-29. Swift 6 패키지, macOS arm64 환경에서 검증한다.
실제 모델·GPU·Jetson·REST/MCP·장치 드라이버를 사용한 통합 테스트는 아니다.

## 회귀 검증 범위

- 정상 업무/MCP/일상 대화, typed heads·masking·STT confidence·partial/final·빈 입력.
- 같은 대상 context, 대상 전환, clarification 연결/만료, 늦은 결과와 context stamp.
- write 성공/인증 오류/응답 유실/timeout/reconciliation, 취소 후 unknown 상태 보존.
- 출력 사실 보호와 fallback, 음성/화면/웹 실패 격리, 전달 완료 context.
- 리마인더 중복·최신 상태 확인·보류·snooze, 큐/요청/context/journal 상한.
- 모듈 제출과 완료 처리의 독립 루프: 완료 처리가 대기 중에도 다음 작업 제출.
- STT/API/장치가 대기 중일 때 다른 모듈의 준비된 작업이 진행됨.
- 단일 GPU 전담 OS 스레드에서 제출과 완료 조회, 요청별 결과 타입/ID의 분리.
- 먼저 제출된 GPU mock이 미완료여도 다음 요청 제출·결과 반환, 오류 격리.
- GPU 취소 시 실제 completion까지 자원/슬롯 유지, shutdown 회수.
- VAD speech 프레임을 end 이전에 STT에 전달, 재생 중 입력 지속, 캡처가 STT 추론을 기다리지 않음.
- bounded PCM 역압/취소, RFC 3339·epoch millisecond·시간대·과대 응답 검사.

이전 구현의 복수 worker/CPU 스레드 개수별 benchmark는 폐기했다.
해당 수치를 현재 모듈 구조나 실제 GPU 성능의 근거로 사용하지 않는다.

## 실행 결과

`swift test --sanitize=thread`로 62개 테스트(파라미터화 사례 포함 64건)가 통과했다.
해당 실행에서 데이터 경합 보고는 없었다. 이 결과가 실제 native runtime의 스레드 안전성을 보장하지는 않는다.
재생 큐 역압력 검증은 장치가 대기 중인 상태에서 화면 6건이 진행하고, 대기를 해제한 뒤 음성 6건이 모두 완료되는지 확인한다.

Release 빌드가 통과했다. `.build/release/deskpet-demo --module-demo` 연결 검증 결과:

| 항목 | 결과 |
| --- | --- |
| 요청 / 전달 완료 / 음성 완료 | 256 / 256 / 256 |
| GPU mock 연산 완료 | 1,024 |
| GPU 전담 OS 스레드 | 1 |
| 남은 진행 요청 / 진단 이벤트 유실 | 0 / 0 |

mock 연결 및 정리 검증이며 실제 GPU 처리량·추론 지연 측정은 아니다.

## 실제 통합에서 남은 검증

- Linux/Jetson의 Swift toolchain·빌드 및 native runtime 연결.
- 추론/출력 모델의 TensorRT 또는 다른 런타임 결정, 비동기 제출/completion handle 계약.
- STT 실행 위치 결정과 한국어 인식률·partial/final·프레임 처리율·최대 backlog.
- TTS 실행 위치 결정, 실제 PCM 포맷과 재생 완료 ack, AEC.
- 모델 weights/workspace/KV cache의 RAM/GPU memory·발열·30분 지속 동작.
- 실제 Eureka response schema/pagination/영속 reconciliation, Harness/MCP/LLM Wiki.
- 장치 명령의 실제 취소/표시 완료 ack, journal/알림 이력의 재시작 영속화.
