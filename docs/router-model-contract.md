# Router Model Output Contract

기준일: 2026-11-01. 기존 Swift Pipeline의 `RouterModel`·`HeadOutputs`·`RouteDecision`과 C++ Backend의 실제 모델 출력 연결 계약이다. Harness 구현·업무 판단·실행 기능 추가는 범위 밖이다.

## Original prediction and execution intent

| 필드 | 의미 |
| --- | --- |
| `modelIntention: Head<ModelIntention>?` | 실제 학습 분류 `eureka`, `llm`, `talk`, `unclear`와 원래 점수. native 결과에는 필수이며 기존 mock의 생략은 호환 지원 |
| `operation: Head<String?>?` | 실제 학습 Operation과 그 점수. `eureka`에서만 활성이고 다른 Intention에서는 null/0 |
| `modelOperation: RouterOperation?` | `operation.value`의 typed view. 별도 예측·점수·직렬화 필드가 아님 |
| `intention: Head<Intent>` | 코드가 구성한 실행 의도. Eureka는 24개 모두 `eureka.<Operation>`. 이 필드의 confidence도 원래 Intention 점수 유지 |
| `target` / `name` / `checkpoint` | 업무(item)·단계(stage)·체크포인트의 원문. API ID나 대상 종류를 추측해 채우지 않음 |
| `task` / `note` | 학습된 생성 head의 이름/요약·설명. 업무 생성 이름은 Task이며 Name을 업무 이름으로 바꾸지 않음 |
| `dueAt` | 공통 코드가 정규화할 수 있는 날짜 또는 null. 원래 표현은 `spans.DueAt`에 보존 |
| `priority` | 학습 Priority 라벨과 점수 |
| `spans` | head별 `{message, start, end, text}`. role 메시지 인덱스와 Unicode scalar 반열린 구간을 보존하고 원문 대조 |
| `modelConfidence` | 실제 Confidence head 결과. `RouteDecision.overallConfidence`로 전달 |
| `generationFinished` | 실행한 Task/Note 생성 head의 종료 여부. 잘린 결과는 실행 요청으로 사용하지 않음 |

C++와 Swift 공통 바인딩은 같은 라벨을 사용한다. Swift는 구형 Backend의 5개 Operation 매핑만 있는 결과도 `modelIntention`·`operation`에서 다시 연결한다. 원본이 `eureka`이면 실행 코드 유무를 이유로 `llm`으로 바꾸지 않는다. 원본이 `llm`·`unclear`이면 그 예측을 그대로 보존한다. 유효하지 않은 Operation은 `ROUTER_OPERATION_UNSUPPORTED`이며 다른 분류로 대체하지 않는다.

`talk`의 실행 의도는 기존 `chat`이다. 실제 pending ID에 대한 명확한 답변은 기존 `confirm.*` 실행 의도로 연결할 수 있지만 `modelIntention`·Operation·원래 점수는 유지한다. 실행 의도는 새로 학습한 head가 아니다.

## All 24 Operations

| 실제 Operation | 실행 의도 |
| --- | --- |
| `create` | `eureka.create` |
| `query` | `eureka.query` |
| `complete` | `eureka.complete` |
| `update_stage` | `eureka.update_stage` |
| `postpone` | `eureka.postpone` |
| `update_priority` | `eureka.update_priority` |
| `update_item` | `eureka.update_item` |
| `hold` | `eureka.hold` |
| `resume` | `eureka.resume` |
| `delete` | `eureka.delete` |
| `start_stage` | `eureka.start_stage` |
| `reset_stage` | `eureka.reset_stage` |
| `skip_stage` | `eureka.skip_stage` |
| `complete_task` | `eureka.complete_task` |
| `query_processes` | `eureka.query_processes` |
| `query_actors` | `eureka.query_actors` |
| `query_workspace` | `eureka.query_workspace` |
| `add_stage` | `eureka.add_stage` |
| `delete_stage` | `eureka.delete_stage` |
| `delete_task` | `eureka.delete_task` |
| `add_task` | `eureka.add_task` |
| `add_note` | `eureka.add_note` |
| `move_stage_up` | `eureka.move_stage_up` |
| `move_stage_down` | `eureka.move_stage_down` |

`query`, `query_processes`, `query_actors`, `query_workspace`는 조회이며 나머지는 변경 동작으로 분류한다. 기존 키보드 읽기 전용 제한·확인·권한·신뢰도 정책을 완화하지 않는다.

## Dispatch and adapter boundary

- 유효한 라벨은 실행 코드 존재 여부와 관계없이 `HeadOutputs`·`ActionRequest`·Eureka Adapter에 표현할 수 있다. 실제 기능 지원은 실행 구현의 책임이다. 구현이 없는 mock은 `EUREKA_OPERATION_NOT_IMPLEMENTED`로 실패하며 성공이나 다른 분류로 위장하지 않는다.
- 낮은 Confidence·STT·모호성에 대한 기존 경로 결정은 유지한다. 이 경우에도 원본 head·Operation은 바꾸지 않는다. 실행 미구현을 이유로 Harness로 보내는 분기는 없다.
- 전역 조회는 업무/단계 ID를 요구하지 않는다. native `create`·`add_task`의 생성 이름은 Task로 받으며 구형 register의 Name 규칙을 적용하지 않는다. 모델의 Target 원문과 이미 확인된 업무 ID는 서로 다른 정보다. 미해결 원문을 PR ID 등으로 대체하지 않고 실행 Adapter에 원래 head와 함께 전달한다.
- Eureka 결과에 이전 PR 리마인드의 `github_pr` 대상을 업무 ID로 재사용하지 않는다. PR→Eureka 전환 시 관련 없는 PR pending과 이전 대화 묶음을 해제한다. 이 변경은 원래 모델 입력을 다른 말로 바꾸거나 재추론하는 기능이 아니다.
- ServerEureka의 구형 wire 호환 이름만 `eureka.create`→`eureka.register`, `eureka.update_priority`→`eureka.priority`로 변환한다. 원본 `modelIntention`·`operation`·Span·점수는 그대로 전달한다. 업무 ID가 실제로 확인된 `query`만 기존 status 조회 이름으로 연결한다. 나머지 Operation은 canonical 이름 그대로 전달한다.
- 호출 전의 명시적인 `KEYBOARD_READ_ONLY`·`EUREKA_OPERATION_NOT_IMPLEMENTED`는 확정 실패이며, 외부 요청 응답 유실의 `unknown`과 구분한다. 결과 불명 쓰기는 기존 계약대로 재전송하지 않는다.

## Console

아래는 형식을 설명하는 예시이며 실측값이 아니다.

```text
모델 원본: Intention=eureka (0.98), Operation=query_actors (0.96)
라우팅: eureka.query_actors → eureka; 종합 0.85, 의도 0.98
```

라벨 연결·인터페이스 갱신을 실제 모델의 분류 품질 개선이나 24개 업무 기능의 구현 완료로 해석하지 않는다. 사용자 지시에 따라 이번 변경에서 테스트·빌드·모델 실행은 수행하지 않는다.

## Router reference context — 2026-11-01

이전 대화는 기본으로 유지하고 원래 user/assistant role·턴 경계와 마지막 user 위치로 현재 발화를 구분한다. system 지시문·추가 텍스트 표식은 학습 입력 규격을 변경하므로 제거했다. `--no-router-context`는 현재 발화만의 결과를 비교할 때만 사용한다. 상세 LLM Wiki 관리는 Harness 책임으로 유지한다. [학습 입력·컨텍스트 및 Span 계약](router-context.md)을 따른다.
