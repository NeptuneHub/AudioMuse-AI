# Screenshot tools

To regenerate the example screenshots into `screenshot/example/`, run:

```
python -m pip install playwright && python -m playwright install chromium

export AUDIOMUSE_BASE="http://YOUR-SERVER:8000"
export AUDIOMUSE_USER="admin"
export AUDIOMUSE_PW="admin"
export AUDIOMUSE_OUT="screenshot/example"

python driver.py && python driver2.py
```
