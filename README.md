# PlutoSDR 위성 수신 분석

PlutoSDR로 받은 위성 IQ를 분석한 기록과 Python 스크립트입니다.

## 수신 기록

- 2026-11-18 AO-123 (435.400 MHz). 예측 도플러를 따라 움직인 후보의 주파수 변화 폭이 보정 후 18.52 kHz에서 146 Hz로 줄었습니다. [전체 기록](docs/observations/2026-11-18-ao123.md)

![AO-123 도플러 보정 전후 워터폴](docs/observations/ao123/02-doppler-comparison.png)

## 분석 스크립트

- `analyze_live_doppler.py`는 도플러 보정 전후의 스펙트럼과 워터폴을 비교합니다.
- `analyze_rs44.py`는 RS-44 비콘의 예측 도플러를 관측 피크와 비교하고 CW 오디오를 추출합니다.
- `decode_meteor.py`는 METEOR LRPT 신호를 SatDump로 후처리합니다.

[실행 방법과 입력 형식](docs/usage.md)

## 라이선스

[MIT License](LICENSE)
