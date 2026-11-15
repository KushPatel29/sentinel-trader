@echo off
cd /d "%~dp0crypto_lab"
python -m sentinel_crypto.cli stop
python -m sentinel_crypto.cli focus-stop
pause
