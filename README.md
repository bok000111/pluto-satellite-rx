# PlutoSDR 위성 수신 분석

PlutoSDR로 받은 위성 IQ를 분석하는 Python 스크립트입니다. 저장된 신호를 이용해 도플러 보정 전후를 비교하고, RS-44 비콘의 CW 오디오를 추출합니다. 실행에는 직접 준비한 녹음 파일을 사용합니다.

## 저장된 수신 결과 분석

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다. `recording.json`, `iq.cs16`, `live-doppler/uncorrected.cf32`, `live-doppler/corrected.cf32`가 있는 녹음 폴더를 지정합니다. IQ 형식은 각각 little-endian int16 I/Q 쌍과 little-endian complex64입니다.

```sh
uv run --locked python scripts/analyze_live_doppler.py path/to/recording
```

`recording.json`의 `sample_count`, `bytes_written`, `target`과 `live_doppler`의 `output_sample_rate_hz`, `output_samples_per_branch`, `processed_samples`, `rejected_samples`, `status`, `errors`를 읽습니다. IQ 크기와 처리된 샘플 수를 대조하고 `live-doppler/`에 스펙트럼, 워터폴, `comparison.json`을 저장합니다. `residual_correction.status`가 `completed`이면 `residual-corrected.cf32`도 비교합니다.

## RS-44 CW 비콘 분석

```sh
uv run --locked python scripts/analyze_rs44.py path/to/recording --prediction path/to/prediction.json
```

녹음 폴더에 `iq.cs16`과 `recording.json`이 필요합니다. 메타데이터에는 `sample_count`, `sample_rate_hz`, `center_frequency_hz`, 시간대가 포함된 ISO 8601 `actual_start_utc`를 넣습니다. 예측 JSON에는 `reference_frequency_hz`에 435605000을, `time_utc_unix`와 `doppler_hz`에 같은 길이의 시간순 배열을 넣습니다. 예측 구간은 녹음을 덮어야 합니다.

예측 도플러에 주파수 편차를 더해 관측 피크와 비교하고, 녹음 폴더에 추정값과 그래프, `cw-candidate.wav`를 저장합니다. PSD의 full scale은 2048 ADC counts입니다. 호출부호는 추출한 오디오에서 별도로 확인합니다.

## 테스트

```sh
PYTHONPATH=scripts uv run --locked python -m unittest discover -s tests
```

테스트는 생성한 신호로 도플러 추정과 CW 오디오 추출를 확인합니다.

## 라이선스

[MIT License](LICENSE)
