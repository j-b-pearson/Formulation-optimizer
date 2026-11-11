"""Save Streamlit app screenshots to reports/v1/figures/app_*.png (headless Chrome or Edge).

Start the app first, in another PowerShell window:
    uv run streamlit run app/streamlit_app.py --server.headless true
Then:
    uv run python scripts/app_screenshots.py

The app's `?demo=1` option replays the demo script (6 initial runs, 5 suggestions, one manual
entry) in each new session, and `&view=...` renders one view without tabs. This script drives
the browser through the Chrome DevTools Protocol (the `websockets` package ships with
Streamlit), waits until each view has finished rendering, and saves a full-page PNG.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]
SHOTS = [
    ("app_optimize_enamel.png", "demo=1&view=optimize", "Campaign history"),
    ("app_benchmark_enamel.png", "view=benchmark", "Paired differences"),
    ("app_price_shock_enamel.png", "demo=1&view=price&ps_pai_resin=25", "Re-optimized"),
    ("app_about.png", "view=about", "What this is"),
]


class Cdp:
    """Minimal synchronous DevTools client: send a command, wait for its reply."""

    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=None, open_timeout=30, legacy=True)
        self.next_id = 0

    def call(self, method: str, **params) -> dict:
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=120))
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str):
        res = self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        return res.get("result", {}).get("value")


def wait_until_rendered(cdp: Cdp, marker: str, timeout: float = 90.0) -> None:
    """Wait for the marker text and for Streamlit's 'running' widget to disappear."""
    deadline = time.time() + timeout
    check = (
        "(() => { const t = document.body ? document.body.innerText : '';"
        f" const ok = t.includes({json.dumps(marker)});"
        " const running = document.querySelector('[data-testid=\"stStatusWidget\"]');"
        " return ok && !running; })()"
    )
    while time.time() < deadline:
        if cdp.eval(check):
            time.sleep(3.0)  # let Plotly finish drawing
            return
        time.sleep(1.0)
    raise TimeoutError(f"Page never showed {marker!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501/")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--port", type=int, default=9333)
    args = parser.parse_args()
    browser = next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        raise SystemExit("Chrome or Edge not found; take the screenshots by hand")
    out_dir = ROOT / "reports" / "v1" / "figures"  # config paths.figures_dir
    profile = Path(tempfile.mkdtemp(prefix="formopt_shots_"))
    proc = subprocess.Popen(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={args.port}",
            f"--user-data-dir={profile}",
            f"--window-size={args.width},1000",
            "about:blank",
        ]
    )
    try:
        targets: list[dict] = []
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json/list") as r:
                    targets = [t for t in json.load(r) if t.get("type") == "page"]
                if targets:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        cdp = Cdp(targets[0]["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call(
            "Emulation.setEmulatedMedia",
            features=[
                {"name": "prefers-color-scheme", "value": "light"},
                {"name": "prefers-reduced-motion", "value": "reduce"},
            ],
        )
        for name, query, marker in SHOTS:
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                width=args.width,
                height=1000,
                deviceScaleFactor=1,
                mobile=False,
            )
            cdp.call("Page.navigate", url=f"{args.url}?{query}")
            wait_until_rendered(cdp, marker)
            height = int(
                cdp.eval(
                    "(() => { const m = document.querySelector("
                    "'[data-testid=\"stMainBlockContainer\"]')"
                    " || document.querySelector('[data-testid=\"stMain\"]');"
                    " return m ? Math.ceil(m.getBoundingClientRect().bottom + 40)"
                    " : document.body.scrollHeight; })()"
                )
                or 1000
            )
            height = min(max(height, 1000), 6000)
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                width=args.width,
                height=height,
                deviceScaleFactor=1,
                mobile=False,
            )
            # Headless Chrome sometimes leaves the sidebar half-collapsed; open it if so.
            cdp.eval(
                "(() => { const b = document.querySelector("
                '\'[data-testid="stExpandSidebarButton"] button, '
                '[data-testid="stSidebarCollapsedControl"] button,'
                ' [data-testid="stExpandSidebarButton"]\');'
                " if (b) { b.click(); return true; } return false; })()"
            )
            time.sleep(2.5)
            shot = cdp.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            path = out_dir / name
            path.write_bytes(base64.b64decode(shot["data"]))
            print(f"Wrote {path} ({height}px tall)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
