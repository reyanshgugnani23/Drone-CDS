# 🛸 Drone-CDS (Drone Detection & Alert System)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Supabase](https://img.shields.io/badge/Supabase-Enabled-emerald?logo=supabase)
![Telegram](https://img.shields.io/badge/Telegram-Bot%20API-blue?logo=telegram)

**Drone-CDS** is an automated computer-vision monitoring and detection system designed for drone feeds and autonomous road inspection. It processes real-time video streams to identify structural road defects (potholes), filters out false positives like road markings, and automatically broadcasts real-time photo alerts to a designated **Telegram** channel or chat.

---

## 🌟 Features

- **Real-Time Visual Detection:** Processes video streams or live camera feeds using OpenCV edge and contour analysis.
- **Smart Filtering:** Intelligent ROI (Region of Interest) mapping to reduce false positives caused by off-road terrain, side barriers, and bright lane markings.
- **Instant Telegram Alerts:** Transmits annotated image snapshots with detection metrics directly to a Telegram bot.
- **Secure Credentials Management:** Integrates with `.env` and Supabase secrets vault to keep API tokens and secret keys secure.
- **Cooldown Safeguard:** Built-in alert rate-limiting to avoid flooding notification streams during continuous detection.

---

## 🛠️ Project Structure

```text
├── main.py              # Main detection pipeline and entry point
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (Bot Token, Chat ID)
├── .gitignore           # Ignores sensitive environment configuration and logs
└── README.md            # Project documentation





