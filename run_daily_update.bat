@echo off

cd /d "C:\Users\Rishab Sawlani\OneDrive\Desktop\StockVision-AI"

call venv\Scripts\activate.bat

python -m src.data.update_market_data >> data\daily_update.log 2>&1