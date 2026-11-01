# Router Role Context

2026-11-01: 상세 컨텍스트·LLM Wiki 관리는 Harness 책임으로 유지한다. 라우터에는 기존에 선별된 이전 role 대화와 현재 사용자 발화를 그대로 전달한다. 컨텍스트를 기본으로 폐기하거나 다른 문장으로 요약·재작성하지 않는다.

## 학습 입력 규격

확인한 학습 코드 `render_messages`·`encode_messages`는 원본 `messages`에 checkpoint의 chat template를 적용한다. 옵션은 `add_generation_prompt=true`, `enable_thinking=false`, 토크나이징은 `add_special_tokens=false`다. 마지막 템플릿 토큰의 hidden state에 분류 head를 연결한다. 현재 구현도 이 경로를 유지한다.

원래 user/assistant role과 순서가 과거 턴을 구분한다. 마지막 user 메시지가 현재 발화다. system 지시문, `[REFERENCE]`·`[CURRENT]` 텍스트, 별도 구분 문장과 요약을 추가하지 않는다. 직전의 잘못된 프레이밍은 활성·비활성 두 모드 모두에서 제거했다.

Backend 입력은 원문 `messages`와 `currentMessageIndex`다. 인덱스는 마지막 메시지를 가리켜야 하며 해당 role은 user여야 한다. 인덱스·설정은 입력 계약 메타데이터이고 모델이 읽는 token에 추가하지 않는다. 메타데이터만으로 모델에 새로운 구분 능력이 생긴다고 가정하지 않는다. 모델에 제공하는 구분 신호는 학습에 사용한 chat-template의 role·턴 경계다.

이전 메시지는 Span 후보에서 제외하지 않는다. 현재 발화에서 생략된 업무·단계 등을 과거 원문에서 참조할 수 있다. Span의 message·Unicode scalar 좌표는 원본 메시지 배열 기준으로 유지한다. Intention·Operation·점수는 모델 실제 출력을 그대로 전달하고 문장 규칙으로 덮어쓰지 않는다.

## 실행

기본은 컨텍스트 유지다. 기존 `Models/native.local.json`의 설정도 유지하며 `routerContextEnabled`가 없으면 true다.

```sh
./run.command --keyboard
```

동일 문제가 이어져 현재 발화만의 결과와 비교할 때 아래 옵션을 사용한다. 이 경우에도 system 지시문이나 표식을 넣지 않는다.

```sh
./run.command --keyboard --no-router-context
```

이 옵션은 라우터에 전달하는 이전 role 메시지만 제외한다. Harness·출력 모델의 컨텍스트, conversation ID·대상 metadata·실제 pending 확인 상태는 유지한다. 자동으로 비활성화하지 않는다.

Swift는 구형 Backend의 잘못된 표식 주입도 피하도록 호환 필드 `contextMode: legacy`를 보낸다. 새 Backend에서는 이 값과 관계없이 원래 템플릿만 사용한다. 첫 실행은 `--no-build` 없이 시작해 C++ 렌더러 수정도 반영한다.

## 남은 확인 범위

입력 규격 복구를 분류 품질 개선의 실측 결과로 보고하지 않는다. 확인한 데이터 생성 설정에는 독립 발화 학습과 후속 대화 세트 조립 계획이 기록되어 있다. 실제 checkpoint의 대화 학습 범위나 GGUF 양자화의 수치적 영향은 이 코드 수정만으로 확정하지 않는다. 사용자 지시에 따라 테스트·빌드·모델 실행은 수행하지 않는다.
