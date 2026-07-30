"""Put the app backend on the import path.

`app/` is a service, not an installed package (unlike `src/`, which pyproject
installs), so its modules need the directory added explicitly before
`backend.analysis.*` can be imported here.
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
