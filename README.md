# PlutoSDR 위성 수신 분석

PlutoSDR로 받은 위성 IQ를 분석하는 Python 스크립트입니다. 저장된 신호를 이용해 도플러 보정 전후를 비교하고, RS-44 비콘의 CW 오디오를 추출하고, METEOR LRPT 신호를 후처리합니다. 실행에는 직접 준비한 녹음 파일을 사용합니다.

[AO-123 수신 기록 (2026-11-18)](docs/observations/2026-11-18-ao123.md)

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

예측 도플러에 주파수 편차를 더해 관측 피크와 비교하고, 녹음 폴더에 추정값과 그래프, `cw-candidate.wav`를 저장합니다. PSD의 full scale은 2048 ADC counts입니다. 호출부호는 추출한 오디오에서 확인합니다.

## METEOR LRPT 후처리

별도로 설치한 [SatDump CLI](https://github.com/SatDump/SatDump/tree/1.2.2#cli-version)를 `PATH`에 두거나 `--satdump`로 실행 파일을 지정합니다. 이 스크립트는 `meteor_m2-x_lrpt`와 `meteor_m2-x_lrpt_80k` 파이프라인을 사용합니다.

```sh
uv run --locked python scripts/decode_meteor.py path/to/recording
```

`live-doppler/corrected.cf32`와 `recording.json`이 필요합니다. 기록 메타데이터의 `status`와 `live_doppler.status`는 `completed`여야 하고, `settings_restored`, `live_doppler.covers_all_recorded_samples`, `live_doppler.file_sizes_match`가 참이어야 합니다. `live_doppler`에는 `output_sample_rate_hz`, `output_samples_per_branch`, `fir_passband_hz`도 필요합니다. 기본 필터를 쓰려면 보존한 한쪽 통과 대역이 80 kHz보다 커야 합니다. `meteor-recovery/`가 이미 있으면 실행을 거부하므로 다시 분석할 때는 새 입력 사본을 사용합니다.

잔여 반송파 보정과 채널 필터를 적용한 분기를 비교합니다. 추정한 심볼레이트를 먼저 시도하고, 미확정이면 72 ksym/s를 먼저 시도합니다. 유효 CADU가 없으면 다른 심볼레이트도 시도합니다. Reed-Solomon 검사를 통과한 CADU를 중복 제거해 `meteor-recovery/merged-rs-accepted.cadu`에 합칩니다. CADU가 검출되면 같은 심볼레이트의 SatDump 파이프라인으로 `merged-products/`에 후처리를 실행합니다. 실행 오류가 없으면 유효 CADU가 0개여도 `completed`로 기록합니다. 분기별 결과와 수용된 CADU 수는 `meteor-recovery/comparison.json`에 기록됩니다.

## 테스트

```sh
PYTHONPATH=scripts uv run --locked python -m unittest discover -s tests
```

테스트는 생성한 신호로 도플러 추정, 채널 필터, 반송파 추정과 CADU 중복 처리를 확인합니다.

## 라이선스

[MIT License](LICENSE)
