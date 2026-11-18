# 실행 방법

## 준비와 패스 예측

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다. `config.example.toml`을 복사하고 `[observer]`의 위도, 경도, 고도와 `[pluto]`의 URI, 선택적 serial 주석을 실제 값으로 채웁니다. `config/local.toml`은 git에서 제외됩니다.

```sh
mkdir -p config
cp config.example.toml config/local.toml
```

장비 수신에는 libiio가 필요합니다. 수신 명령에는 `uv run --locked --extra receive`를, 예측과 저장된 결과 분석에는 `uv run --locked`를 사용합니다.

이름 행이 선택적으로 포함된 2행 또는 3행 TLE 파일로 예측합니다. 시작과 끝은 시간대가 포함된 ISO 8601이며, 수신에 쓸 예측 창은 녹음 종료보다 2초 이상 뒤까지 포함해야 합니다.

```sh
uv run --locked python scripts/predict_pass.py --tle path/to/tle.txt --start 2026-11-18T00:00:00Z --end 2026-11-19T00:00:00Z --frequency 435400000 --output path/to/prediction.json
```

`--min-elevation`의 기본값은 10도이고 `--config`의 기본값은 `config/local.toml`입니다. JSON에는 `reference_frequency_hz`, Unix 초 배열 `time_utc_unix`, Hz 배열 `doppler_hz`와 `passes`가 있습니다. Doppler는 2초 간격으로 표본화하며 마지막 끝점도 포함합니다. 각 pass에는 `aos_utc`, `peak_utc`, `los_utc`, `peak_elevation_deg`가 있습니다.

## 수신

`--plan`에 JSON 계획을 지정합니다. `--config` 기본값은 `config/local.toml`이며 `output`과 `prediction`의 상대 경로는 계획 파일을 기준으로 합니다. 주파수, 대역폭, 표본율과 cutoff는 Hz, `seconds`는 초이며 IQ는 interleaved little-endian int16 I/Q 표본으로 저장됩니다. 이미 존재하는 출력은 덮어쓰지 않습니다.

즉시 raw-only 수신 계획 예시입니다.

```json
{
  "target": "example",
  "center_hz": 435400000,
  "sample_rate_hz": 2500000,
  "rf_bandwidth_hz": 1000000,
  "output": "capture",
  "seconds": 30
}
```

예약 수신은 `seconds` 대신 시간대가 포함된 `record_start`와 `record_end`를 사용합니다. 시작 15분 전부터 종료 전까지 실행할 수 있습니다. 다음은 예측 JSON을 사용하는 단일 paired 채널 예시입니다.

```json
{
  "target": "example",
  "center_hz": 435400000,
  "sample_rate_hz": 3000000,
  "rf_bandwidth_hz": 1000000,
  "output": "capture",
  "record_start": "2026-11-18T12:00:00Z",
  "record_end": "2026-11-18T12:10:00Z",
  "prediction": "prediction.json",
  "doppler_channels": [
    {"key": "channel", "carrier_hz": 435400000, "sample_rate_hz": 100000, "cutoff_hz": 10000}
  ]
}
```

```sh
uv run --locked --extra receive python scripts/record_pass.py --plan path/to/plan.json
```

채널은 입력 표본율을 출력 표본율로 정수 배 decimation해야 하며 선택적으로 `cfo_hz`를 지정할 수 있습니다. 채널이 하나면 루트에 `live_doppler` 결과가 있고, 여러 개면 고유한 `key`마다 `targets/<key>` view와 루트 `live_doppler_channels`가 만들어집니다. `gain_control_mode`는 선택 사항입니다. 수신은 최대 15분입니다.

## 저장된 수신 결과 분석

`recording.json`, `iq.cs16`, `live-doppler/uncorrected.cf32`, `live-doppler/corrected.cf32`가 있는 녹음 폴더를 지정합니다. IQ 형식은 각각 little-endian int16 I/Q 쌍과 little-endian complex64입니다.

```sh
uv run --locked python scripts/analyze_live_doppler.py path/to/recording
```

`recording.json`의 `sample_count`, `bytes_written`, `target`과 `live_doppler`의 `output_sample_rate_hz`, `output_samples_per_branch`, `processed_samples`, `rejected_samples`, `status`, `errors`를 읽습니다. `residual_correction.status`가 `completed`이면 `residual-corrected.cf32`도 비교합니다.

## AO-123 후보와 NFM 분석

먼저 위의 `analyze_live_doppler.py`를 실행해 스펙트럼을 만든 뒤 후보를 대조합니다.

```sh
uv run --locked python scripts/analyze_ao123.py --recording path/to/recording --prediction path/to/prediction.json --output path/to/ao-analysis
```

01부터 03까지의 그림과 `candidate-report.json`이 생성됩니다. 04에는 보정 전후 전체 WAV가 모두 필요합니다. 아래는 각 분기에서 WAV를 만들고 함께 넘기는 예시입니다.

```sh
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-before --branch uncorrected --offset-hz 0
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-after --branch corrected --offset-hz 208
uv run --locked python scripts/analyze_ao123.py --recording path/to/recording --prediction path/to/prediction.json --output path/to/ao-with-audio --before-wav path/to/nfm-before/uncorrected-nfm.wav --after-wav path/to/nfm-after/corrected-nfm.wav
```

05와 06은 시간대가 포함된 ISO 시각으로 구성한 비어 있지 않은 windows JSON 배열을 사용합니다. `strong_start`부터 `strong_end`까지는 `start`와 `end` 사이의 0.05초 이상 구간이며, 각 창 양끝에 실제 IQ 0.5초가 더 있어야 합니다.

06의 시작을 정하려면 배열에서 창 하나에만 `example_start`를 지정합니다. 이 시각부터 1초가 해당 창 안에 들어와야 합니다. 생략하면 strong 구간의 RF 피크 대비가 가장 높은 창에서 `strong_start`부터 최대 1초를 사용합니다. `example_rf_peak_excess_db`는 선택된 창의 strong 구간에서 측정한 값입니다.

```json
[
  {"start": "2026-11-18T12:01:00Z", "end": "2026-11-18T12:01:05Z", "strong_start": "2026-11-18T12:01:02Z", "strong_end": "2026-11-18T12:01:03Z", "example_start": "2026-11-18T12:01:03Z"}
]
```

```sh
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-windows --offset-hz 208 --windows path/to/windows.json
```

입력은 100 kS/s paired IQ이며 기본 분기는 `corrected`입니다. `--offset-hz`는 선택한 분기에 적용할 잔류 오프셋입니다.

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
