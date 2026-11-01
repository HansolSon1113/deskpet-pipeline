# 로컬 모델

가중치와 로컬 실행 설정은 Git에서 제외한다. `native.example.json`을 `native.local.json`으로 복사해 사용한다. 경로는 설정 파일이 있는 폴더 기준이다. 인증값은 여기에 넣지 않고 기존 `LLM_test/.env`를 사용한다.

`tools/fetch-native-models.sh`는 고정 revision의 공개 모델을 준비한다. 기존 학습 체크포인트는 다음 위치에 보존해야 한다.

```text
router/heads.pt
router/router_config.json
router/split_indices.json
router/backbone/adapter_model.safetensors
router/backbone/adapter_config.json
router/tokenizer/tokenizer.json
router/tokenizer/tokenizer_config.json
router/tokenizer/chat_template.jinja
```

`tools/prepare-models.sh`는 LibTorch C++ 준비 도구로 다음 산출물을 만든다. 원본 파일은 덮어쓰지 않는다. 이미 있는 산출물은 재사용한 뒤 검사한다. 중간 파일은 `.part`로 쓰고 성공 후 이동한다.

| 산출물 | 용도 |
| --- | --- |
| `router/reference/model.safetensors`, `config.json` | 원본 좌표계에 LoRA를 반영한 비양자화 기준. 향후 TensorRT 변환의 출발점 |
| `router/backbone-merged-bf16.gguf` | 로컬 backend의 양자화 전 비교 기준 |
| `router/backbone-merged-q4_0.gguf` | 로컬 라우터 INT4 backbone |
| `router/heads-f16.gguf` | 28개 학습 head 텐서, FP16 |
| `emotion/model.onnx` | KoELECTRA-Small NSMC의 CPU ONNX 그래프 |
| `prepared-manifest.json`, `prepared.sha256` | 출처·revision·입력 제한·정밀도·라벨·원본/산출물 체크섬 |

GGUF Q4_0은 혼합 정밀도 형식이다. 현재 라우터에는 Q4_0 텐서 194개, Q6_K embedding 1개, FP32 텐서 140개가 있다. head는 모두 FP16이다. `deskpet-model-prep inspect`로 실제 구성을 확인한다.

모든 head 텐서의 합은 약 1.02 GB이다. 이 값에 backbone·KV/recurrent cache·작업 공간·출력 모델·음성 모델 메모리가 추가되므로, 전체 라우터가 1.2 GB에 들어간다고 판단하지 않는다.

실행 절차와 남은 검증은 [네이티브 모델 연결](../docs/native-models.md)을 참고한다.
