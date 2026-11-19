# 실행 방법

## 준비와 패스 예측

필수: Python 3.13, [uv](https://docs.astral.sh/uv/). 장비 수신에는 libiio 필요.

`config.example.toml`을 복사한 뒤 실제 값 입력.

- `[observer]`: 위도, 경도, 고도
- `[pluto]`: URI, serial 주석(선택)
- `config/local.toml`: git 추적 제외

```sh
mkdir -p config
cp config.example.toml config/local.toml
```

- 수신: `uv run --locked --extra receive`
- 예측, 저장된 결과 분석: `uv run --locked`

패스 예측 입력:

- TLE 파일: 2행 또는 3행(이름 행 선택)
- 시작과 끝: 시간대가 포함된 ISO 8601
- 수신용 예측 창: 녹음 종료보다 2초 이상 뒤까지 포함

```sh
uv run --locked python scripts/predict_pass.py --tle path/to/tle.txt --start 2026-11-18T00:00:00Z --end 2026-11-19T00:00:00Z --frequency 435400000 --output path/to/prediction.json
```

기본값: `--min-elevation` 10도, `--config`에 `config/local.toml`.

예측 JSON:

- `reference_frequency_hz`
- `time_utc_unix`(Unix 초 배열), `doppler_hz`(Hz 배열): 2초 간격, 마지막 끝점 포함
- `passes`: 각 pass에 `aos_utc`, `peak_utc`, `los_utc`, `peak_elevation_deg`

## 수신

- `--plan`: JSON 계획. `output`과 `prediction`의 상대 경로는 계획 파일 기준.
- `--config` 기본값: `config/local.toml`
- 단위: 주파수, 대역폭, 표본율, cutoff는 Hz. `seconds`는 초.
- IQ 저장 형식: interleaved little-endian int16 I/Q 표본
- 기존 출력 덮어쓰기 없음.

즉시 raw-only 수신 계획

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

예약 수신은 `seconds` 대신 `record_start`와 `record_end` 사용(시간대 포함). 실행 가능 시점은 시작 15분 전부터 종료 전까지.

예측 JSON과 단일 paired 채널을 쓰는 예약 수신 계획

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

- 채널: 입력 표본율에서 출력 표본율로 정수 배 decimation. `cfo_hz` 선택.
- 채널 하나: 루트에 `live_doppler` 결과
- 채널 여러 개: 고유한 `key`마다 `targets/<key>` view, 루트에 `live_doppler_channels`
- `gain_control_mode`: 선택
- 수신: 최대 15분

## 저장된 수신 결과 분석

녹음 폴더에 필요한 파일:

- `recording.json`
- `iq.cs16`: little-endian int16 I/Q 쌍
- `live-doppler/uncorrected.cf32`, `live-doppler/corrected.cf32`: little-endian complex64

```sh
uv run --locked python scripts/analyze_live_doppler.py path/to/recording
```

읽는 메타데이터:

- `recording.json`: `sample_count`, `bytes_written`, `target`
- `live_doppler`: `output_sample_rate_hz`, `output_samples_per_branch`, `processed_samples`, `rejected_samples`, `status`, `errors`

`residual_correction.status`가 `completed`이면 `residual-corrected.cf32`도 비교.

## AO-123 후보와 NFM 분석

`analyze_live_doppler.py`로 스펙트럼 생성 후 후보 대조.

```sh
uv run --locked python scripts/analyze_ao123.py --recording path/to/recording --prediction path/to/prediction.json --output path/to/ao-analysis
```

출력: 그림 01–03, `candidate-report.json`.

그림 04: 보정 전후 전체 WAV 필요. 각 분기에서 WAV를 만든 뒤 함께 전달.

```sh
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-before --branch uncorrected --offset-hz 0
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-after --branch corrected --offset-hz 208
uv run --locked python scripts/analyze_ao123.py --recording path/to/recording --prediction path/to/prediction.json --output path/to/ao-with-audio --before-wav path/to/nfm-before/uncorrected-nfm.wav --after-wav path/to/nfm-after/corrected-nfm.wav
```

그림 05와 06의 windows JSON:

- 비어 있지 않은 배열. 시각은 시간대를 포함한 ISO 형식.
- `strong_start`부터 `strong_end`까지는 `start`와 `end` 사이의 0.05초 이상 구간.
- 각 창 양끝에 실제 IQ 0.5초씩 필요.

그림 06 시작 시각:

- `example_start`는 배열의 창 하나에만 지정. 지정 시각부터 1초가 해당 창 안에 포함되어야 함.
- 생략 시 strong 구간의 RF 피크 대비가 가장 높은 창에서 `strong_start`부터 최대 1초 사용.
- `example_rf_peak_excess_db`: 선택한 창의 strong 구간에서 측정한 값.

```json
[
  {"start": "2026-11-18T12:01:00Z", "end": "2026-11-18T12:01:05Z", "strong_start": "2026-11-18T12:01:02Z", "strong_end": "2026-11-18T12:01:03Z", "example_start": "2026-11-18T12:01:03Z"}
]
```

```sh
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-windows --offset-hz 208 --windows path/to/windows.json
```

- 입력: 100 kS/s paired IQ
- 기본 분기: `corrected`
- `--offset-hz`: 선택한 분기에 적용할 잔류 오프셋

## RS-44 CW 비콘 분석

```sh
uv run --locked python scripts/analyze_rs44.py path/to/recording --prediction path/to/prediction.json
```

- 녹음 폴더: `iq.cs16`, `recording.json`
- 메타데이터: `sample_count`, `sample_rate_hz`, `center_frequency_hz`, `actual_start_utc`(시간대를 포함한 ISO 8601)
- 예측 JSON: `reference_frequency_hz`에 435605000, `time_utc_unix`와 `doppler_hz`에 같은 길이의 시간순 배열.
- 예측 구간: 전체 녹음 구간 포함.

## METEOR LRPT 후처리

별도 설치: [SatDump CLI](https://github.com/SatDump/SatDump/tree/1.2.2#cli-version). `PATH`에 두거나 `--satdump`로 실행 파일 지정.

파이프라인: `meteor_m2-x_lrpt`, `meteor_m2-x_lrpt_80k`.

```sh
uv run --locked python scripts/decode_meteor.py path/to/recording
```

- 입력: `live-doppler/corrected.cf32`, `recording.json`
- `completed` 필수: `status`, `live_doppler.status`
- 참 필수: `settings_restored`, `live_doppler.covers_all_recorded_samples`, `live_doppler.file_sizes_match`
- `live_doppler` 필수 필드: `output_sample_rate_hz`, `output_samples_per_branch`, `fir_passband_hz`
- 기본 필터: 보존한 한쪽 통과 대역이 80 kHz 초과.
- `meteor-recovery/`가 이미 있으면 실행 불가. 재분석은 새 입력 사본 사용.

## 테스트

```sh
PYTHONPATH=scripts uv run --locked python -m unittest discover -s tests
```
