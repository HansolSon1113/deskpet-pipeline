# 네이티브 모델 연결

기존 Swift 역할 모듈과 Adapter에 모델을 연결한다. 업무 판단·Gateway·모듈 스케줄러는 기존 구현을 사용한다. 위키는 수정하지 않는다.

## Backend 경계

```text
RouterModel / OutputModel
  → NativeInferenceModel
  → GPUOperation.submit → GPUCompletion.poll
  → C ABI plugin
      llama.cpp / ggml / Metal
```

공개 C ABI는 `Sources/CNativeInference/include/deskpet_native.h`에 있다. 모델 설정과 입력·결과 JSON, opaque 모델·job handle만 전달한다. Swift 공통 타입에는 llama context·ggml tensor·CUDA stream·TensorRT 객체를 넣지 않는다.

`submit`은 즉시 handle을 반환한다. llama.cpp의 blocking decode는 backend가 소유한 bounded worker에서 수행하고, 기존 GPU 전담 스레드는 completion을 nonblocking으로 확인한다. 모델별 worker는 동일 context의 동시 사용을 막는 실행 자원이다. 새 우선순위·GPU scheduler는 추가하지 않는다. 호출 취소는 native job에 전달되며, 물리 완료 전에는 job·모델·라이브러리와 GPU admission 슬롯을 해제하지 않는다.

설정의 `backend`와 `device`를 명시적으로 검사한다. 라우터·출력은 `metal`/`cuda`만 허용한다. token embedding까지 GPU buffer로 지정하며, llama graph가 CPU 수치 연산에 배치되면 제출 전 `GPU_OPERATION_UNSUPPORTED`로 실패한다. head graph도 모든 연산의 GPU 지원을 검사한다. 초기화·추론 실패 시 다른 backend나 CPU로 전환하지 않는다.

TensorRT 전환 시 같은 `deskpet_backend_v1` ABI를 구현한 plugin을 만들고 해당 모델의 `library`와 `backend`를 교체한다. 예를 들어 라우터와 출력의 `library`만 TensorRT plugin으로 지정하고 CPU 음성 모델은 기존 library를 사용할 수 있다. Swift 역할 모듈은 그대로 둔다. TensorRT 엔진·환경별 빌드는 이번 구현에 포함하지 않는다. 원본 safetensors, LoRA, head checkpoint, tokenizer, 설정과 LoRA 반영 기준 safetensors를 보존한다. TensorRT 내부에서 backbone/head를 합치거나 나누는 방식은 강제하지 않는다.

## 모델 계약

| 역할 | 구현 | 배치 |
| --- | --- | --- |
| VAD | Silero VAD ONNX / sherpa-onnx | CPU |
| STT | 다국어 Whisper small Q8_0 / whisper.cpp | CPU |
| 라우터 | Qwen3.5 0.8B + 학습 head | GPU, backbone Q4_0·head FP16 |
| 출력 | Qwen3.5 2B Q4_0 | GPU |
| 감정 | KoELECTRA-Small NSMC / ONNX Runtime C++ | CPU |
| TTS | Supertonic 3 INT8 / sherpa-onnx C API | CPU, 한국어 |

라우터는 학습 tokenizer의 chat template와 role 메시지를 사용한다. 마지막 hidden state의 분류, token+pooled hidden의 Span, Task·Note 생성 head, Intention 마스킹 뒤 Confidence head를 실행한다. `modelIntention`은 학습 분류를, `operation`은 학습 Operation을 그대로 보존한다. `intention`은 Eureka 24개 라벨 모두 `eureka.<Operation>`으로 일대일 연결한다. 예를 들어 `eureka + query_actors`는 `eureka.query_actors`이며 실행 코드 유무를 이유로 `llm`으로 바꾸지 않는다. Task는 업무 이름/요약이고 Name은 단계 원문이다. Operation·Checkpoint·Span 원문과 Unicode scalar 좌표·생성 종료 여부를 보존한다. Span은 원문과 다시 대조한다. 생성이 한도에서 잘리면 실행하지 않는다.

최신 필드·24개 라벨·구형 wire 별칭의 경계는 [라우터 모델 출력 계약](router-model-contract.md)을 따른다. Swift 공통 바인딩도 실제 `modelIntention`·`operation`을 기준으로 실행 의도를 구성하므로 구형 Backend plugin의 불완전한 매핑이 결과를 `llm`으로 바꿔도 복원한다. 콘솔에는 원본 Intention·Operation·각 점수와 최종 경로를 별도로 표시한다. 모델 원본이 `llm`·`unclear`인 경우에는 그 결과를 유지한다.

명확한 pending 답변의 실행 의도만 공통 코드에서 `confirm.*`로 연결한다. 학습 분류와 원래 Confidence는 보존한다. `llm`은 현재 발화·대상·최근 role 메시지와 함께 기존 Harness로 간다. 제공된 데모 PR은 PR을 가리키는 발화에만 적용한다. 출력 모델은 일반 대화를 생성하며, 승인 질문과 업무 결과는 기존 고정 문장을 그대로 전달한다.

DueAt 원문은 Span에 항상 남긴다. ISO 날짜와 `오늘/내일/모레`만 코드에서 정규화하고, 해석이 모호한 표현은 추가 확인을 받는다. 감정은 확정된 사용자 발화에 적용하며, 긍정/부정 확률이 0.8 이상인 경우에만 표정을 바꾸고 나머지는 중립으로 둔다. 업무 상태의 표정이 우선한다.

STT 입력은 16 kHz mono PCM16, 최대 30초이며 1초마다 partial과 VAD end에서 final을 만든다. 실제 Whisper token probability로 점수를 계산하며 final만 라우팅한다. 부분 입력은 업무 쓰기 승인을 만들지 않는다. Supertonic은 한 번의 native 생성으로 PCM을 만든 뒤 bounded chunk로 `AudioSource`에 전달한다. 현재 TTS는 문장 전체가 준비된 뒤 첫 chunk를 제공한다.

Silero가 음성을 확정하기 전의 첫 자음·PR 번호를 보존하도록 Mac 입력에는 최대 200 ms의 pre-roll을 둔다. 크기와 프레임 수를 제한하고 epoch가 바뀌면 폐기한다.

Mac은 AVAudioEngine 마이크/스피커를 사용한다. 캡처를 유지하면서 재생 중과 종료 후 300 ms에는 모델 입력 전달을 억제한다. epoch 변경은 VAD/STT 세션을 초기화하며 오래된 final을 거절한다. 스피커의 `finish`는 `.dataPlayedBack` 완료를 기다린다. Hardware adapter도 현재 SDK의 input epoch를 전달한다.

## 준비와 빌드

Apple Silicon Mac에서 CMake, Swift, C++ compiler와 OpenMP 라이브러리가 필요하다. LibTorch는 준비 도구에만 링크한다. 추론 plugin에는 링크하지 않는다.

```sh
sh tools/setup-native-deps.sh
sh tools/fetch-native-models.sh
sh tools/build-native.sh -DDESKPET_PREP=ON -DCMAKE_PREFIX_PATH="$PWD/.build/native-deps/libtorch"
cmake --build .build/native --target deskpet-model-prep -j 4
sh tools/prepare-models.sh
swift build
cp Models/native.example.json Models/native.local.json
```

OpenMP 위치가 다르면 `DESKPET_OPENMP_LIB`로 디렉터리를 지정한다. 다른 플랫폼의 SDK는 `SHERPA_ROOT`와 CMake의 `CMAKE_PREFIX_PATH`로 지정한다. GPU graph guard는 고정 llama.cpp checkout에만 적용되며 patch 지점이 달라지면 빌드를 중단한다.

CPU만 점검할 수 있다. 합성 음성 점검은 실제 마이크 인식률 평가를 대신하지 않는다.

```sh
.build/debug/deskpet-demo --native Models/native.local.json --probe-cpu
.build/debug/deskpet-demo --native Models/native.local.json --speech-check 'PR 1번을 검토했어요. 승인할까요?' --recognize-generated --wav .build/speech.wav
.build/debug/deskpet-demo --native Models/native.local.json --recognize-pcm .build/recording-16k-mono-s16le.pcm
```

## 실제 음성 실행

하나의 진입점으로 실행하려면 pipeline 폴더에서 다음 명령을 사용한다. macOS Finder에서 `run.command`를 더블클릭해도 같은 실행이 시작된다.

```sh
./run.command
```

Swift 실행 파일, 네이티브 backend와 하드웨어 모의 화면을 증분 빌드하고 기존 Harness 의존성이 없으면 설치한다. macOS에서는 Homebrew를 통해 누락된 SDL2·SDL2_ttf를 설치하고 CMake에 각 설치 경로를 전달한다. 모의 화면 → Harness 인증 연결 준비 → 실제 음성 파이프라인 순서로 시작한다. 외부 API 테스트나 모델 재변환은 실행 단계에 포함하지 않는다. 쓰기 capability는 열지만 실제 변경은 기존 확인 흐름을 따른다.

`Ctrl-C` 또는 구성 요소의 비정상 종료 시 이 진입점이 띄운 프로세스만 종료한다. 다른 실행과 충돌하면 오류를 알리며 기존 프로세스를 종료하거나 임의로 재사용하지 않는다. 서비스 로그는 실행별 `.build/run/` 폴더에 저장한다. 기존 `.env`의 비밀값을 명령 인자로 넘기지 않는다.

| 옵션 | 동작 |
| --- | --- |
| `--keyboard` | 마이크·STT 대신 터미널 문장 입력. 확정 입력 신뢰도 1, 승인·업무 변경 전달 가능. Enter로 전송, `/quit` 또는 Ctrl-C로 종료 |
| `--direct-llm` | 라우터 로드·추론 없이 기존 Harness의 Gemini로 최종 발화 전달 |
| `--metrics-dir DIR` | 실행별 성능 CSV를 저장할 상위 폴더 지정. 기본 `.build/performance` |
| `--check` | 모델·설정 파일을 점검하고 종료. GPU·마이크·외부 API·서비스 실행 없음 |
| `--no-build` | Swift 진입점 외의 증분 빌드를 생략하고 준비된 구성 요소 실행 |
| `--config FILE` | 다른 native 설정 파일 사용 |
| `--headless` | SDL 화면 기능을 빌드하지 않고 하드웨어 모의 창 없이 실행 |
| `--read-only` | Harness의 GitHub·Eureka 쓰기 capability를 열지 않음 |

```sh
./run.command --check
./run.command --no-build
```

키보드 입력은 `./run.command --keyboard` 또는 `run-keyboard.command` 더블클릭으로 시작한다. 기본 리마인드와 기존 대화 맥락·라우터·Harness·TTS/스피커 경로를 사용하며 마이크를 시작하거나 STT·VAD 모델을 로드하지 않는다. 한 문장씩 입력한 뒤 Enter로 전송한다. `/quit`, `/exit`, EOF 또는 Ctrl-C로 종료하고 이 진입점이 시작한 프로세스를 정리한다.

키보드와 `--text`는 사용자 지시에 따라 확정 입력 신뢰도 1을 전달한다. 키보드의 전달용 종합 신뢰도도 1로 두며 원본 라우터 head·modelConfidence는 그대로 보존한다. 키보드 전용 읽기 차단과 파이프라인의 별도 승인 신뢰도 조건을 제거했다. 기본 진입점은 키보드에서도 서버의 쓰기 capability를 열며, `--read-only`를 명시한 경우에만 끈다. 확인 질문·사용자 답변·pending ID 등 기존 Harness 업무 흐름은 유지하고 Harness 구현은 수정하지 않는다.

라우터를 생략하려면 `--direct-llm`을 추가한다. 라우터 모델을 로드하지 않고 모든 최종 입력을 기존 Harness의 Gemini로 전달하며 전달용 신뢰도는 1이다. 모델별 소요 시간과 프로세스 메모리는 `.build/performance/`에 기록한다. [모델 성능 기록](model-performance.md)에 파일 형식과 측정 범위를 설명한다.

기본 저장소 위치는 pipeline의 형제 폴더 `LLM_test`, `deskpet-hardware`다. 위치가 다르면 `DESKPET_HARNESS_DIR`, `DESKPET_HARDWARE_DIR` 환경 변수로 지정한다. 필요한 개발 도구는 Swift·CMake·Node.js·pnpm·Homebrew이며, 모델 최초 준비는 위 절차를 따른다. 화면 실행에 필요한 SDL2·SDL2_ttf는 진입점이 준비한다.

각 구성 요소를 별도로 실행하는 절차는 아래와 같다.

기존 `LLM_test/.env`에 `HARNESS_WS_TOKEN`, `GITHUB_TOKEN`, `GITHUB_TEST_REPO`, `GITHUB_TEST_PR`, `DEMO_EUREKA_ITEM_ID`, `DEMO_EUREKA_STAGE_ID`, 기존 Gemini 설정을 둔다. 인증값을 저장소에 커밋하지 않는다.

기존 하드웨어 저장소에서 모의 화면을 실행한다.

```sh
cmake -S . -B build -DDESKPET_LINUX=OFF
cmake --build build -j 2
./build/deskpet-hardware --config config/mock.json
```

기존 LLM_test 폴더에서 서버를 실행한다. 쓰기 허용 옵션은 capability만 열며 실제 변경은 각각의 확인 질문과 음성 답변을 요구한다.

```sh
pnpm harness --allow-github-writes --allow-eureka-writes
```

pipeline 폴더에서 실행한다. `hardwareSocket: "auto"`는 현재 SDK의 기본 socket을 사용한다. 경로를 직접 지정할 수도 있다. `hardwareSocket`을 생략하면 진단용 MockDisplay가 사용된다.

```sh
.build/debug/deskpet-demo --native Models/native.local.json --probe
.build/debug/deskpet-demo --native Models/native.local.json --say '안녕하세요. 짧게 인사해 주세요.'
.build/debug/deskpet-demo --native Models/native.local.json --live
```

마이크 접근을 허용하면 DeskPet이 먼저 `PR 1번 검토를 잊지 마세요. 변경 내용을 확인해 드릴까요?`라고 리마인드한다. 이 시작 안내는 설정된 PR을 사용하는 시나리오 리마인드이며 업무의 실제 마감이나 상태를 만들어 넣지 않는다. 기존 출력·TTS·화면·재생 완료 경로를 사용하고, 전달 완료 후에만 PR 대상과 안내 문장을 대화 맥락에 저장한다. 사용자 발화나 승인 답변을 합성하지 않는다.

리마인드 청취 → `그거 리뷰해줘` → `승인해줘` → 실제 확인 질문 청취 → `네` → Eureka 반영 질문 청취 → `네` 순서로 진행한다. GitHub 승인과 Eureka 완료는 별도 pending ID·output ID로 처리한다. 실제 재생 완료 또는 확인된 화면 표시 후에만 `question_delivered`를 보고한다. `Ctrl-C`는 입력을 취소하고 종료까지 기다린 뒤 단계별 지연과 process peak memory를 출력한다.

시작 리마인드는 `--live`에서 기본으로 켜진다. 기존 `native.local.json`에도 별도 수정 없이 적용된다. `startupReminder: false`로 끄거나 `startupReminderText`로 안내 문장을 지정할 수 있다. `--text` 등 진단 실행에서는 시작 리마인드를 하지 않는다.

네이티브 데모의 일반 요청은 라우팅·STT 신뢰도 기준을 우선 `0.3`으로 완화한다. `Models/native.local.json`의 최상위 `routeConfidenceThreshold`·`sttConfidenceThreshold`로 각각 조절하며, 필드가 없는 기존 설정에도 `0.3`이 적용된다. 실제 점수는 변경하지 않는다. 최종 STT 점수·종합/의도 점수·선택 경로·로컬 재질문 사유를 콘솔에 표시한다. 누락 슬롯 처리와 실제 승인 답변의 STT `0.85`·라우터 `0.8` 기준은 유지한다.

라우터의 `llm` 분류는 그대로 전달한다. Harness의 LLM Guide가 현재 발화와 최근 대화로 `prepare_pr_approval`을 선택하면, Harness 내부에서만 `pr.approve` 목표로 해석하고 기존 승인 규칙에 연결한다. 콘솔의 `하네스 판단`·`하네스 재질문 사유`로 구분한다. 모델 출력·원래 점수는 바꾸지 않으며, 실제 pending ID가 없는 로컬 재질문은 승인 동의로 취급하지 않는다. Harness도 최초 승인 요청부터 STT `0.85`·라우터 `0.8`을 독립 검사하므로, 일반 요청 기준을 넘더라도 이 기준보다 낮으면 재질문한다. 별도 확인 질문과 전달 완료·답변 검증 전에는 승인을 실행하지 않는다.

`--text`는 진단용이며 STT confidence를 0으로 전달하므로 음성 승인을 대신하지 않는다. 현재 입력 score 0.85 이상, router score 0.8 이상인 명확한 답변만 `mutationApproved`를 전달하며, 최종 범위·만료·중복·새 커밋·권한 검사는 기존 Harness가 수행한다.

## 검증

```sh
swift test
# LLM_test에서 pnpm web:fixture를 켠 뒤 실행
DESKPET_SERVER_TESTS=1 swift test
```

네이티브 ABI fixture는 실제 GPU 없이 초기화 실패, 고정 질문 보존, 취소 후 물리 완료까지의 슬롯 수명을 검증한다. 질문 미전달·두 단계 확인·만료·지연 recovery notice에 대한 회귀 검증도 포함한다. notice는 기존 request를 갱신하고 결과를 출력하며 외부 작업을 다시 실행하지 않는다. 더 최신 turn이 있으면 오래된 notice 출력은 억제한다. 연결이 끊겨도 미확인 쓰기 요청을 자동 재전송하지 않는다.

준비 도구의 `audit-heads`는 FP16 cast 결과의 바이트 일치, `audit-tokenizer`는 vocabulary ID 일치, `inspect`는 GGUF 텐서 타입·크기·파일 경계를 검사한다. `audit-emotion`은 ONNX 결과를 LibTorch C++ 기준 계산과 비교한다. 출처와 SHA256은 `Models/prepared-manifest.json`에 기록한다.

양자화 전후 동일 test split은 사용자 GPU 환경에서 비교한다. 아래 명령은 순서대로 BF16 모델과 Q4_0 모델을 로드해 head 결과와 Confidence 변화를 저장한다. `--limit`을 생략하면 test split 전체를 사용한다.

```sh
mkdir -p out
.build/debug/deskpet-demo --native Models/native.local.json \
  --evaluate-router /path/to/original/data.jsonl \
  --split Models/router/split_indices.json \
  --reference Models/router/backbone-merged-bf16.gguf \
  --report out/router-quantization.json --limit 16
```

모델 연결과 통합 진입점 구현을 이번 작업의 완료 범위로 한다. 실제 GPU·마이크 업무 흐름·GPU 메모리/지연·양자화 전후 head 품질과 외부 업무 검사는 사용자가 수행한다. 점검 항목은 한국어·PR 번호·승인/거절/취소·잡음/무음, 새 커밋, 응답 유실, 부분 실패, TTS 실패와 실제 PR/Eureka 상태의 대조다. CPU fallback은 제공하지 않는다. TensorRT 엔진과 Jetson 실물 성능은 후속 범위다.

## Router reference context — 2026-11-01

이전 대화는 기본으로 유지하고 원래 user/assistant role·턴 경계와 마지막 user 위치로 현재 발화를 구분한다. system 지시문·추가 텍스트 표식은 학습 입력 규격을 변경하므로 제거했다. `--no-router-context`는 현재 발화만의 결과를 비교할 때만 사용한다. 상세 LLM Wiki 관리는 Harness 책임으로 유지한다. [학습 입력·컨텍스트 및 Span 계약](router-context.md)을 따른다.
