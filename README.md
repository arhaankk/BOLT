# Bolt⚡️

## Get Started

### On MacOS:
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload 
pytest -v --disable-warnings 
```

### On Windows:
```bash
python3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest -v --disable-warnings
```
