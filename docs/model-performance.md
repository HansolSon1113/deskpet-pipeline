# 모델 성능 기록과 LLM 직결

기존 진입점은 실행할 때마다 `.build/performance/<UTC 시각>-<router 또는 direct-llm>-<실행 ID>/`에 성능 파일을 만든다. 콘솔에는 측정 행이나 JSON을 출력하지 않는다. 다음 명령은 사용자가 실행할 진입점이며 구현 과정에서 빌드·테스트·모델 실행은 수행하지 않았다.

```sh
# 기존 라우터 경로
./run.command --keyboard

# 라우터 로드·추론 없이 모든 최종 발화를 기존 Harness의 Gemini로 전달
./run.command --keyboard --direct-llm

# 마이크 입력에서도 동일한 옵션 사용
./run.command --direct-llm

# 저장할 상위 폴더 지정: 이 안에 실행별 하위 폴더 생성
./run.command --keyboard --direct-llm --metrics-dir /tmp/deskpet-comparison
```

`run-keyboard.command`, `run-linux.sh`, `run-linux-keyboard.sh`에서도 동일한 옵션을 전달할 수 있다. 직접 `deskpet-demo --native CONFIG` / `--jetson-native CONFIG`를 실행할 때도 사용할 수 있다. `Models` 실행 설정의 `directLLM: true`, `performanceDirectory: "../.build/performance"`로 지정할 수도 있다. 설정의 상대 기록 경로는 설정 파일 기준이며 CLI 경로는 실행 디렉터리 기준이다.

| 파일 | 기록 |
| --- | --- |
| `run.json` | 라우터/직결 모드, 키보드 여부, 라우터 컨텍스트 여부, 모델 Backend·장치·파일·토큰 제한·스레드, OS·CPU 수·장치 물리 메모리 |
| `calls.csv` | 호출 ID, 실행 시작 후 시각, 모델 역할, Backend, 장치, 단계, 성공/실패/취소, 소요 ms, 호출 전후 프로세스 RSS와 증감, 호출 구간의 샘플 RSS/footprint 최대값, 프로세스 수명 전체 RSS 최대값 |
| `memory.csv` | 100ms 간격의 프로세스 RSS·footprint·수명 전체 RSS 최대값·활성 호출 수 |
| `summary.csv` | 종료 시 모델·Backend·장치·단계별 호출 수, 실패/취소 수, 전체 평균·최대, 최근 최대 1,024회 p50/p95, 관측 메모리 최대값 |
| `recording-status.json` | 종료 시 기록 누락 수와 파일 쓰기 오류. 디스크가 밀리면 최대 512행만 대기시키고 누락을 명시 |

VAD·STT·라우터·출력·감정·TTS의 **실제로 실행한 호출**만 기록한다. 로드는 `load`, 일반 추론은 `inference`, STT는 `partial`/`final`, VAD 초기화는 `reset`으로 분리한다. 키보드 모드는 STT/VAD 모델을 로드하지 않고, 직결 모드는 라우터를 로드하지 않는다. 출력 모델은 기존 Adapter 구성을 유지하며 실제 생성이 필요할 때만 추론한다. 시작 리마인드의 감정/TTS 호출도 실제 호출이므로 기록에 포함한다.

네이티브 소요 시간은 입력 직렬화부터 완료 polling까지의 호출 벽시계 시간이며 GPU 큐 대기 시간도 포함한다. 모델 결과의 콘솔 출력 시간은 제외한다. 네이티브 성공/실패는 Backend 완료 결과 기준이다. `harness / remote-llm / remote / roundtrip`은 기존 Swift Harness Adapter 왕복 시간으로 네트워크·원격 LLM·도구 실행이 함께 포함된다. Gemini만의 추론 시간이나 원격 서버 메모리로 해석하면 안 된다. Harness 구현에는 측정 코드를 추가하지 않았다.

메모리는 **파이프라인 프로세스 전체** 기준이다. 각 모델 로드 전후 차이와 호출 구간을 비교할 수 있지만 동시 실행·공유 메모리·할당 캐시 때문에 모델 단독 점유량이 아니다. Mac은 RSS와 `phys_footprint`, Linux는 RSS를 기록한다. Linux에서 지원하지 않는 footprint와 조회 실패 값은 CSV 빈칸으로 남긴다. 100ms 샘플은 짧은 순간 최대값을 놓칠 수 있고, 수명 전체 RSS 최대값은 호출별 최대값과 다르다. 이 값들을 모델별 GPU VRAM이나 원격 Gemini 메모리로 표시하거나 합산하지 않는다. 동일 입력·모델 설정·장치에서 워밍업 전후를 나누어 비교한다.

직결은 기존 Pipeline의 입출력·컨텍스트·ID·완료 처리를 유지하고 라우터 단계에서 `.harness`로 목적지만 지정한다. 사용자 원문과 이전 role 메시지, 활성 대상, pending ID, taskRefs는 기존 Harness 요청 계약으로 전달한다. 사용자 지시에 따라 직결 경로의 head/wire 신뢰도는 `1`로 설정하며 `routingMode: "direct-llm"`, `confidenceSource: "configured-direct-llm"`으로 명시한다. 이는 선택된 연결 경로의 설정값이며 모델의 예측 점수가 아니다. `modelIntention`·`modelConfidence`는 없다.

키보드와 `--text`의 확정 입력 신뢰도는 `1`이다. 키보드의 RouteDecision 종합 신뢰도도 `1`, `confidenceSource: "confirmed-keyboard-input"`으로 전달한다. 라우터가 실제 산출한 head·modelConfidence와 콘솔의 원본 점수는 변경하지 않는다. 키보드 전용 읽기 차단과 파이프라인의 별도 승인 신뢰도 차단을 제거했다. 기본 실행은 키보드에서도 기존 서버의 GitHub/Eureka 쓰기 옵션을 켜며, 사용자가 명시적으로 `--read-only`를 선택한 경우에만 서버 쓰기를 끈다. 확인 질문·실제 답변·pending ID·전달 완료 등 기존 업무 흐름은 Harness 계약을 그대로 사용하며 Harness 구현은 수정하지 않는다.
