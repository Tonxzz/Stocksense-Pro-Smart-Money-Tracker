# StockSense Pro v2.1 - Smart Money Screener

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🎯 Overview

**StockSense Pro** is an institutional-grade Smart Money Flow Screener for Indonesian Stock Exchange (IDX). It uses advanced technical indicators to detect accumulation patterns and volume anomalies that may indicate institutional activity.

## ✨ Features

- 🔍 **Sector-Based Screener**: Scan 10 sectors with 20-40 stocks each
- 📊 **AI Scoring System**: 0-5 star rating based on RVOL, CMF, MFI, VWAP
- 📈 **4-Panel Technical Chart**: Price + SMA 200 + VWAP, RSI, CMF Flow, RVOL
- 🛡️ **Safety Filters**: Auto-exclude penny stocks and illiquid tickers
- 🌙 **Dark Mode UI**: Professional institutional-grade interface

## 🧠 Smart Money Logic (v2.1)

| Indicator | Purpose | Scoring |
|-----------|---------|---------|
| **RVOL > 1.5** | Volume Explosion | +2 ⭐ |
| **CMF > 0.15** | Strong Accumulation | +2 ⭐ |
| **CMF > 0.05** | Moderate Inflow | +1 ⭐ |
| **Price > VWAP** | Trend Confirmation | +1 ⭐ |
| **MFI < 30 + Uptrend** | Strategic Dip Buy | +1 ⭐ |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app_pro.py
```

## 📁 Project Structure

```
├── app_pro.py            # Main Streamlit application
├── stocksense_engine.py  # Backend engine (data + indicators)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## 📝 License

MIT License - Free for personal and commercial use.
