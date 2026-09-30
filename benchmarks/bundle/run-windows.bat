@echo off
cd /d "%~dp0"
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt || goto :err
set M=models
python bench.py %M%\nano-s112-fp32.onnx %M%\nano-s112-fp16.onnx %M%\fast-s224-fp32.onnx %M%\fast-s224-fp16.onnx %M%\large-l224-fp32.onnx %M%\large-l224-fp16.onnx %M%\incumbent\rotation_estimator_fp32.onnx --providers cuda --batch 64 --runs 200 --out bench-results.jsonl
python bench.py %M%\nano-s112-int8.onnx %M%\fast-s224-int8.onnx %M%\large-l224-int8.onnx %M%\incumbent\rotation_estimator_fp32.onnx --providers cpu --threads 1,4 --runs 60 --out bench-results.jsonl
echo Done: bench-results.jsonl
goto :eof
:err
echo Setup failed. Is Python 3.10-3.12 installed and on PATH?
