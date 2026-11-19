# PlutoSDR 위성 수신 분석

PlutoSDR로 받은 위성 IQ를 분석한 기록과 Python 스크립트.

## 수신 기록

- 2026-11-19 RS-44 (435.605 MHz). 도플러 편차 −2,550 Hz, 피크 33개 잔차 RMS 51 Hz. CW에서 `RS44` 두 번 판독. [전체 기록](docs/observations/2026-11-19-rs44.md)
- 2026-11-18 AO-123 (435.400 MHz). 예측 도플러를 따른 후보의 주파수 변화 폭: 보정 전 18.52 kHz, 보정 후 146 Hz. [전체 기록](docs/observations/2026-11-18-ao123.md)

![AO-123 도플러 보정 전후 워터폴](docs/observations/ao123/02-doppler-comparison.png)

[수신과 분석 실행 방법](docs/usage.md)

## 라이선스

[MIT License](LICENSE)
