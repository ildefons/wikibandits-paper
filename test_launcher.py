"""Exercise GPU_on_Windows.ipynb on Linux with fake wsl.exe / powershell.exe / nvidia-smi.
Run with a Python that has IPython + notebook: /tmp/wbvenv/bin/python test_launcher.py"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NB = nbformat.read(HERE / "GPU_on_Windows.ipynb", as_version=4)
CODE = [c.source for c in NB.cells if c.cell_type == "code"]
FAKE_BIN = "/tmp/fakewin/bin"
PATCHES = [('IS_WINDOWS = os.name == "nt"', "IS_WINDOWS = True"),
           ('VENV = "/opt/wikibandits-gpu"', 'VENV = "/tmp/wbvenv"'),
           ("HERE = Path.cwd()", 'HERE = Path("C:/Users/Ildefons/wb test")'),
           ('STATE = Path.home() / ".wikibandits_gpu.json"', 'STATE = Path("state.json")'),
           ("NO_WINDOW = 0x08000000 if IS_WINDOWS else 0", "NO_WINDOW = 0"),            # Linux rejects these flags
           ("DETACHED = (0x00000200 | NO_WINDOW) if IS_WINDOWS else 0", "DETACHED = 0")]


def patched(src):
    for a, b in PATCHES:
        if a in src:
            src = src.replace(a, b)
    return src


def scenario(name, flags, cells=None, env=None, root=None, pre=None):
    root = root or Path(tempfile.mkdtemp(prefix="wb_"))
    state = root / "fake_state"
    state.mkdir(exist_ok=True)
    for f in flags:
        (state / f).touch()
    folder = root / "C:" / "Users" / "Ildefons" / "wb test"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "WikiBandits.ipynb", folder / "WikiBandits.ipynb")
    os.chdir(root)
    os.environ.update(FAKE_STATE=str(state), FAKE_ROOT=str(root), BROWSER="true",
                      PATH=FAKE_BIN + ":" + os.environ["PATH"].replace(FAKE_BIN + ":", ""))
    os.environ.pop("FAKE_DRIVER", None)
    os.environ.update(env or {})
    ns = {}
    stopped = None
    print(f"\n===== {name} =====")
    for i, src in enumerate(CODE[: cells or len(CODE)]):
        if pre and i in pre:
            src = pre[i](src)
        try:
            exec(compile(patched(src), f"cell{i}", "exec"), ns)
        except Exception as e:
            if type(e).__name__ == "Stop":
                stopped = str(e)
                print("".join(e._render_traceback_()))
                break
            raise
    calls = (state / "calls.log").read_text() if (state / "calls.log").exists() else ""
    return ns, stopped, calls, root


def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as r:
        return r.status, r.read()


ok = []

# 1. Fresh PC: WSL missing, install works without a restart; full run starts a real Jupyter.
ns, stopped, calls, root = scenario("fresh install", ["install_works"])
assert stopped is None, stopped
assert "'--install','-d','Ubuntu','--no-launch'" in calls, calls
assert ns["CUDA"] == "cuda13" and ns["DISTRO"] == "Ubuntu"
token, host, port = ns["token"], ns["host"], ns["PORT"]
st, body = get(f"http://{host}:{port}/api/contents/WikiBandits.ipynb?token={token}")
assert st == 200 and json.loads(body)["name"] == "WikiBandits.ipynb"
st, body = get(ns["links"][0][0])
assert st == 200 and b"<html" in body.lower()
assert ns["wsl_path"]("C:\\Users\\Ildefons\\Documents\\wb test") == "/mnt/c/Users/Ildefons/Documents/wb test"
assert ns["wsl_path"]("D:\\") == "/mnt/d/"
assert ns["decode"]("Ubuntu\r\n".encode("utf-16-le")) == "Ubuntu\n"
ok.append("fresh install -> Jupyter served the notebook from the Windows folder")

# 2. Run all again while it is running: reuses the server.
ns2, stopped, calls, _ = scenario("run again", [], root=root)
assert stopped is None and ns2["token"] == token, stopped
ok.append("second Run all reuses the running GPU Jupyter")

# 3. Stop cell.
ns3, stopped, calls, _ = scenario("stop", [], root=root,
                                  pre={len(CODE) - 1: lambda s: s.replace("STOP_GPU_JUPYTER = False", "STOP_GPU_JUPYTER = True")})
time.sleep(2)
assert ns3["server_host"]() is None, "server still up"
assert not (root / "state.json").exists()
ok.append("stop cell shuts the GPU Jupyter down")

# 4. WSL needs a restart to finish.
_, stopped, _, _ = scenario("restart needed", [], cells=3)
assert stopped and "restart" in stopped.lower(), stopped
ok.append("asks for a restart when WSL isn't ready")

# 5. User clicks No on the permission prompt.
_, stopped, _, _ = scenario("permission refused", ["cancel"], cells=3)
assert stopped and "refused" in stopped, stopped
ok.append("explains a refused permission prompt")

# 6. CUDA 13 doesn't work: falls back to CUDA 12.
ns, stopped, calls, _ = scenario("cuda13 broken", ["installed", "cuda13_broken"], cells=4)
assert stopped is None and ns["CUDA"] == "cuda12" and ns["backend"] == "GPU", (stopped, ns.get("CUDA"))
assert calls.count("apt-get update") == 2
ok.append("falls back from CUDA 13 to CUDA 12")

# 7. Older driver: goes straight to CUDA 12.
ns, stopped, calls, _ = scenario("driver 560", ["installed"], cells=4, env={"FAKE_DRIVER": "560.94"})
assert stopped is None and ns["CUDA"] == "cuda12" and calls.count("apt-get update") == 1
ok.append("driver 560 -> CUDA 12 directly")

# 8. Nothing gives a GPU: clear message with details.
_, stopped, _, _ = scenario("no gpu", ["installed", "cuda13_broken", "cuda12_broken"], cells=4)
assert stopped and "doesn't see the GPU" in stopped and "Unable to initialize" in stopped, stopped
ok.append("explains when JAX can't see the GPU")

# 9. WSL1 gets converted.
_, stopped, calls, _ = scenario("wsl1", ["installed", "wsl1"], cells=3)
assert "--set-version Ubuntu 2" in calls
ok.append("converts WSL1 to WSL2")

# 10. Too-old driver.
_, stopped, _, _ = scenario("driver 470", [], cells=2, env={"FAKE_DRIVER": "471.11"})
assert stopped and "too old" in stopped
ok.append("rejects drivers older than 525")

print("\nALL PASSED:")
for line in ok:
    print("  -", line)
