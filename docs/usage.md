# 실행 방법

## 준비와 패스 예측

Python 3.13과 [uv](https://docs.astral.sh/uv/) 필요. PlutoSDR로 수신할 때는 libiio도 필요.

`config.example.toml`을 복사한 뒤 실제 값 입력.

- `[observer]`: 위도, 경도, 고도
- `[pluto]`: URI 또는 serial 입력(선택)
- `config/local.toml`: git 추적 제외

```sh
mkdir -p config
cp config.example.toml config/local.toml
```

- 수신: `uv run --locked --extra receive`
- 예측, 저장된 결과 분석: `uv run --locked`

TLE는 2행 또는 이름을 포함한 3행. 시작과 끝은 시간대가 포함된 ISO 8601. 수신용 예측은 녹음 종료보다 2초 이상 뒤까지 계산.

```sh
uv run --locked python scripts/predict_pass.py --tle path/to/tle.txt --start 2000-01-01T00:00:00Z --end 2000-01-02T00:00:00Z --frequency 435400000 --output path/to/prediction.json
```

기본 설정 파일은 `config/local.toml`.

예측 JSON에 기준 주파수, 2초 간격 도플러, 패스 정보 저장.

## 수신

- `--plan`: JSON 계획. `output`과 `prediction`의 상대 경로는 계획 파일 기준.
- `--config` 기본값: `config/local.toml`
- 단위: 주파수, 대역폭, 샘플레이트, cutoff는 Hz. `seconds`는 초.
- IQ 저장 형식: interleaved little-endian int16 I/Q 샘플
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

채널 하나의 도플러 보정 전후 IQ를 저장하는 예약 예시

```json
{
  "target": "example",
  "center_hz": 435400000,
  "sample_rate_hz": 3000000,
  "rf_bandwidth_hz": 1000000,
  "output": "capture",
  "record_start": "2000-01-01T12:00:00Z",
  "record_end": "2000-01-01T12:10:00Z",
  "prediction": "prediction.json",
  "doppler_channels": [
    {"key": "channel", "carrier_hz": 435400000, "sample_rate_hz": 100000, "cutoff_hz": 10000}
  ]
}
```

```sh
uv run --locked --extra receive python scripts/record_pass.py --plan path/to/plan.json
```

- 채널: 입력 샘플레이트를 정수 배율로 decimation. `cfo_hz` 선택.
- 채널 하나: 녹음 폴더에 `live_doppler` 결과
- 채널 여러 개: `targets/<key>`에 채널별 결과, 녹음 폴더에 `live_doppler_channels`
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

`analyze_live_doppler.py`로 스펙트럼을 만든 뒤 도플러 보정 전후의 후보를 비교.

```sh
uv run --locked python scripts/analyze_ao123.py --recording path/to/recording --prediction path/to/prediction.json --output path/to/ao-analysis
```

출력: 그림 01–03, `candidate-report.json`.

그림 04: 보정 전후 전체 WAV 필요. 두 WAV를 분석기에 전달.

```sh
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-before --branch uncorrected --offset-hz 0
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-after --branch corrected --offset-hz 208
uv run --locked python scripts/analyze_ao123.py --recording path/to/recording --prediction path/to/prediction.json --output path/to/ao-with-audio --before-wav path/to/nfm-before/uncorrected-nfm.wav --after-wav path/to/nfm-after/corrected-nfm.wav
```

그림 05와 06의 windows JSON:

- 비어 있지 않은 배열. 시각은 시간대를 포함한 ISO 형식.
- `strong_start`부터 `strong_end`까지는 `start`와 `end` 사이의 0.05초 이상 구간.
- 각 window의 앞뒤로 IQ가 0.5초씩 필요.

그림 06의 `example_start`는 window 하나에만 지정. 지정 시점부터 1초가 같은 window에 들어가야 함. 생략하면 RF peak가 배경보다 가장 높은 `strong_start`부터 최대 1초 사용.

```json
[
  {"start": "2000-01-01T12:01:00Z", "end": "2000-01-01T12:01:05Z", "strong_start": "2000-01-01T12:01:02Z", "strong_end": "2000-01-01T12:01:03Z", "example_start": "2000-01-01T12:01:03Z"}
]
```

```sh
uv run --locked python scripts/demod_nfm.py --recording path/to/recording --output path/to/nfm-windows --offset-hz 208 --windows path/to/windows.json
```

- 입력: 100 kS/s IQ(도플러 보정 전후)
- 기본 branch(`--branch`): `corrected`
- `--offset-hz`: 선택한 branch에 적용할 남은 오프셋(Hz)

## RS-44 CW 비콘 분석

```sh
uv run --locked python scripts/analyze_rs44.py path/to/recording --prediction path/to/prediction.json
```

- 녹음 폴더: `iq.cs16`, `recording.json`
- 메타데이터: `sample_count`, `sample_rate_hz`, `center_frequency_hz`, `actual_start_utc`(시간대를 포함한 ISO 8601)
- 예측 JSON: `reference_frequency_hz`에 435605000, `time_utc_unix`와 `doppler_hz`에 같은 길이의 시간순 배열.
- 예측 구간: 전체 녹음 구간 포함.

저장된 스펙트럼과 CW 오디오로 관측 그림 생성. `--audio-start`는 녹음 시작 후 오디오 클립이 시작하는 지점(초).

```sh
uv run --locked python scripts/analyze_rs44.py path/to/recording --prediction path/to/prediction.json \
  --output path/to/figures --wide-spectrum path/to/wide.npz \
  --cw-audio path/to/cw.wav --audio-start 0
```

## METEOR LRPT 후처리

별도 설치: [SatDump CLI](https://github.com/SatDump/SatDump/tree/1.2.2#cli-version). `PATH`에 두거나 `--satdump`로 실행 파일 지정.

파이프라인: `meteor_m2-x_lrpt`, `meteor_m2-x_lrpt_80k`.

```sh
uv run --locked python scripts/decode_meteor.py path/to/recording
```

- 입력: `live-doppler/corrected.cf32`, `recording.json`
- `completed` 필수: `status`, `live_doppler.status`
- `true` 필수: `settings_restored`, `live_doppler.covers_all_recorded_samples`, `live_doppler.file_sizes_match`
- `live_doppler` 필수 필드: `output_sample_rate_hz`, `output_samples_per_branch`, `fir_passband_hz`
- 기본 필터: 한쪽 passband 80 kHz 초과.
- `meteor-recovery/`가 이미 있으면 실행 불가. 재분석은 새 입력 사본 사용.

## 테스트

```sh
PYTHONPATH=scripts uv run --locked python -m unittest discover -s tests
```
