# 실행 방법

## 저장된 수신 결과 분석

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다. `recording.json`, `iq.cs16`, `live-doppler/uncorrected.cf32`, `live-doppler/corrected.cf32`가 있는 녹음 폴더를 지정합니다. IQ 형식은 각각 little-endian int16 I/Q 쌍과 little-endian complex64입니다.

```sh
uv run --locked python scripts/analyze_live_doppler.py path/to/recording
```

`recording.json`의 `sample_count`, `bytes_written`, `target`과 `live_doppler`의 `output_sample_rate_hz`, `output_samples_per_branch`, `processed_samples`, `rejected_samples`, `status`, `errors`를 읽습니다. `residual_correction.status`가 `completed`이면 `residual-corrected.cf32`도 비교합니다.

## RS-44 CW 비콘 분석

```sh
uv run --locked python scripts/analyze_rs44.py path/to/recording --prediction path/to/prediction.json
```

녹음 폴더에 `iq.cs16`과 `recording.json`이 필요합니다. 메타데이터에는 `sample_count`, `sample_rate_hz`, `center_frequency_hz`, 시간대가 포함된 ISO 8601 `actual_start_utc`를 넣습니다. 예측 JSON에는 `reference_frequency_hz`에 435605000을, `time_utc_unix`와 `doppler_hz`에 같은 길이의 시간순 배열을 넣습니다. 예측 구간은 녹음을 덮어야 합니다.

## METEOR LRPT 후처리

별도로 설치한 [SatDump CLI](https://github.com/SatDump/SatDump/tree/1.2.2#cli-version)를 `PATH`에 두거나 `--satdump`로 실행 파일을 지정합니다. 이 스크립트는 `meteor_m2-x_lrpt`와 `meteor_m2-x_lrpt_80k` 파이프라인을 사용합니다.

```sh
uv run --locked python scripts/decode_meteor.py path/to/recording
```

`live-doppler/corrected.cf32`와 `recording.json`이 필요합니다. 기록 메타데이터의 `status`와 `live_doppler.status`는 `completed`여야 하고, `settings_restored`, `live_doppler.covers_all_recorded_samples`, `live_doppler.file_sizes_match`가 참이어야 합니다. `live_doppler`에는 `output_sample_rate_hz`, `output_samples_per_branch`, `fir_passband_hz`도 필요합니다. 기본 필터를 쓰려면 보존한 한쪽 통과 대역이 80 kHz보다 커야 합니다. `meteor-recovery/`가 이미 있으면 실행을 거부하므로 다시 분석할 때는 새 입력 사본을 사용합니다.

## 테스트

```sh
PYTHONPATH=scripts uv run --locked python -m unittest discover -s tests
```
