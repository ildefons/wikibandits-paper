"""Assemble GPU_on_Windows.ipynb: sets up WSL2 + GPU JAX from a normal Windows Jupyter
and opens WikiBandits in a Jupyter that runs inside WSL on the GPU."""
import nbformat as nbf

INTRO = r"""# Run WikiBandits on your own NVIDIA GPU (Windows)

JAX can only use an NVIDIA GPU from Linux. Windows has a built-in Linux (WSL2).
This notebook sets it up **from here**, then opens `WikiBandits.ipynb` in a second
Jupyter that runs inside that Linux, on your GPU. You keep working in your browser
as now; the notebooks and results stay in this same Windows folder.

**How to use it**

1. Put this file in the same folder as `WikiBandits.ipynb`.
2. Open it in your usual Jupyter (Anaconda) and *Run all*.
3. The first time, follow the messages:
   - Windows asks *"Do you want to allow this app to make changes to your device?"*: click **Yes**;
   - you may have to **restart the PC**; afterwards open this notebook again and *Run all*. It continues where it stopped;
   - it downloads about 3–4 GB (Ubuntu, Python, the GPU libraries): roughly 15–40 minutes, depending on your connection.
4. At the end a new browser tab opens `WikiBandits.ipynb` running on the GPU.

**Next times:** *Run all* here (about 20 seconds), then use the link.

**Two things to remember**

- If `WikiBandits.ipynb` is also open in this Jupyter, close it here (*File → Close and Shut Down Notebook*),
  so two Jupyters don't overwrite each other's saves.
- While runs go, don't let the PC sleep (*Settings → System → Power → Sleep: Never*). The screen can switch off.

Nothing here changes your Anaconda installation: everything is installed inside Ubuntu.
"""

SETTINGS = r"""# Settings and helpers (you normally don't need to change anything here)
PORT = 8899                    # where the GPU Jupyter listens on this PC
DISTRO = "Ubuntu"              # the Linux that WSL2 runs
VENV = "/opt/wikibandits-gpu"  # Python environment inside Ubuntu
CUDA = "auto"                  # "auto" picks from your driver; or "cuda13" / "cuda12"

import json, os, re, secrets, shlex, shutil, subprocess, time, urllib.parse, urllib.request, webbrowser
from pathlib import Path
from IPython.display import HTML, display

IS_WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0                # don't flash black windows
DETACHED = (0x00000200 | NO_WINDOW) if IS_WINDOWS else 0   # keeps running if this Jupyter closes
ENV = dict(os.environ, WSL_UTF8="1")                       # ask wsl.exe for plain UTF-8 text
HERE = Path.cwd()
STATE = Path.home() / ".wikibandits_gpu.json"


class Stop(Exception):
    # Ends 'Run all' with a short message instead of a long error report.
    def _render_traceback_(self):
        return ["", ">>> " + str(self).replace("\n", "\n    ")]


def decode(raw):
    # Older wsl.exe versions answer in UTF-16.
    text = raw.decode("utf-16-le" if raw.count(0) > len(raw) // 4 else "utf-8", "replace")
    return text.replace("\x00", "").replace("﻿", "").replace("\r", "")


def run(cmd, timeout=900, show=False):
    # Run a Windows program; return (exit code, output). show=True prints the output as it comes.
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=ENV, creationflags=NO_WINDOW)
    except FileNotFoundError:
        return 127, f"{cmd[0]} was not found"
    if not show:
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            return 124, f"{cmd[0]} took longer than {timeout} s"
        return p.returncode, decode(out)
    lines = []
    for raw in iter(p.stdout.readline, b""):
        line = decode(raw).rstrip()
        lines.append(line)
        if line.strip():
            print("   ", line, flush=True)
    return p.wait(), "\n".join(lines)


def wsl_args(script):
    # A one-line bash script without double quotes or backslashes passes through Windows unchanged.
    assert not set(script) & set('"\\\n'), script
    return ["wsl.exe", "-d", DISTRO, "-u", "root", "-e", "bash", "-c", script]


def wsl(script, **kw):
    # Run a bash command as root inside Ubuntu.
    return run(wsl_args(script), **kw)


def as_admin(*args):
    # Run wsl.exe with administrator rights: Windows asks "allow this app to make changes?".
    arglist = ",".join(f"'{a}'" for a in args)
    ps = (f"$p = Start-Process -FilePath wsl.exe -ArgumentList {arglist} -Verb RunAs -Wait -PassThru; "
          "exit $p.ExitCode")
    code, out = run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], timeout=3600)
    if "cancel" in out.lower():
        raise Stop("Windows asked for permission and it was refused. Run this cell again and click Yes.")
    return code, out


print("This folder:", HERE)
"""

STEP1 = r"""# Step 1: Windows and the NVIDIA driver
if not IS_WINDOWS:
    raise Stop("This notebook is for Windows only. On Colab, Linux or inside WSL, open WikiBandits.ipynb directly.")
code, out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
if code != 0:
    raise Stop("Windows can't find the NVIDIA driver (nvidia-smi didn't run). Install the latest GeForce "
               "driver from nvidia.com, restart, then Run all again.\n" + out)
gpu, driver, memory = [s.strip() for s in out.strip().splitlines()[0].split(",")[:3]]
major = int(driver.split(".")[0])
print(f"GPU: {gpu}, {memory}; Windows driver {driver}")
if major < 525:
    raise Stop(f"Driver {driver} is too old for JAX (it needs 525 or newer). "
               "Update it from nvidia.com, restart, then Run all again.")
if CUDA == "auto":
    CUDA = "cuda13" if major >= 580 else "cuda12"
print("JAX will use", CUDA)
"""

STEP2 = r"""# Step 2: WSL2 with Ubuntu (first time: an 'allow changes?' prompt and maybe a restart)
def find_ubuntu():
    code, out = run(["wsl.exe", "-l", "-q"], timeout=120)
    names = [n.strip() for n in out.splitlines() if n.strip()] if code == 0 else []
    if DISTRO in names:
        return DISTRO
    return next((n for n in names if n.lower().startswith("ubuntu")), None)


found = find_ubuntu()
if found is None:
    print("Installing WSL2 and Ubuntu.")
    print("Windows will ask 'Do you want to allow this app to make changes to your device?': click Yes.")
    print("A window then shows the progress (5-15 minutes). Wait until it closes by itself.")
    code, out = as_admin("--install", "-d", "Ubuntu", "--no-launch")
    print(f"The installer finished (exit code {code}).")
    found = find_ubuntu()
    if found is None and shutil.which("ubuntu.exe"):   # older Windows: register Ubuntu without asking for a user
        run([shutil.which("ubuntu.exe"), "install", "--root"], timeout=1800, show=True)
        found = find_ubuntu()
    if found is None:
        raise Stop("WSL is installed, but Windows must restart to finish. Restart the PC, open this notebook "
                   "again and Run all; it continues from here.")
DISTRO = found

code, out = wsl("uname -r", timeout=300)
if code != 0:
    if "0x80370102" in out or "irtuali" in out:     # 'virtualization' / 'virtualización'
        raise Stop("Ubuntu can't start because virtualization is switched off in the PC's BIOS/UEFI "
                   "(look for Intel VT-x, AMD-V or SVM). Switch it on, then Run all again.\n" + out)
    raise Stop("Ubuntu doesn't start yet. If you haven't restarted since the installation, restart and "
               "Run all again. Otherwise send me this message:\n" + out)
if "microsoft-standard" not in out.lower() and "wsl2" not in out.lower():
    print("Ubuntu runs on the older WSL1; switching it to WSL2 (a few minutes)...")
    run(["wsl.exe", "--set-version", DISTRO, "2"], timeout=3600, show=True)
    code, out = wsl("uname -r", timeout=300)
    if "microsoft-standard" not in out.lower() and "wsl2" not in out.lower():
        raise Stop("Couldn't switch Ubuntu to WSL2. Restart the PC and Run all again; if it repeats, "
                   "virtualization may be off in the BIOS/UEFI.\n" + out)
print(f"{DISTRO} runs on WSL2 (kernel {out.strip()})")

SMI = "/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
code, out = wsl(SMI)
if code != 0:
    print("Ubuntu can't see the GPU yet; updating WSL (Windows may ask for permission again)...")
    as_admin("--update")
    run(["wsl.exe", "--shutdown"])
    code, out = wsl(SMI)
    if code != 0:
        raise Stop("Ubuntu still can't see the GPU. Update the NVIDIA driver from nvidia.com, restart, "
                   "and Run all again.\n" + out)
print("GPU visible inside Ubuntu:", out.strip())
"""

STEP3 = r"""# Step 3: Python, JAX and Jupyter inside Ubuntu (first time: ~3 GB download, 5-30 minutes)
CHECK = (f"{VENV}/bin/python -c 'import jax, optax, pandas, matplotlib, notebook; "
         "print(jax.default_backend().upper(), jax.__version__, jax.devices())'")


def jax_status():
    code, out = wsl(CHECK, timeout=600)
    m = re.search(r"^(GPU|CPU) (\S+) (.*)$", out, re.M)
    return (m.group(1), m.group(2), m.group(3)) if code == 0 and m else (None, None, out)


def install(cuda):
    print(f"Installing Python, JAX ({cuda}) and Jupyter inside Ubuntu. Quiet stretches are downloads.")
    steps = ["export DEBIAN_FRONTEND=noninteractive",
             "apt-get update -qq",
             "apt-get install -y -qq python3-venv python3-pip",
             f"(test -x {VENV}/bin/python || python3 -m venv {VENV})",
             f"{VENV}/bin/pip install --upgrade pip",
             f"{VENV}/bin/pip uninstall -y -q jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt",
             f"{VENV}/bin/pip install --upgrade 'jax[{cuda}]' optax numpy pandas matplotlib notebook ipykernel"]
    code, out = wsl(" && ".join(steps), show=True)
    if code != 0:
        raise Stop("The installation stopped with an error (see above). Check the internet connection and run "
                   "this cell again; it picks up where it stopped.")


backend, version, devices = jax_status()
if backend != "GPU":
    install(CUDA)
    backend, version, devices = jax_status()
if backend != "GPU" and CUDA == "cuda13":
    print("JAX doesn't see the GPU with CUDA 13; trying CUDA 12 instead...")
    CUDA = "cuda12"
    install(CUDA)
    backend, version, devices = jax_status()
if backend != "GPU":
    code, why = wsl(f"JAX_PLATFORMS=cuda {VENV}/bin/python -c 'import jax; print(jax.devices())'")
    raise Stop("JAX is installed but doesn't see the GPU. Updating the NVIDIA driver usually fixes it. "
               "Details:\n" + "\n".join(why.strip().splitlines()[-8:]))
print(f"Ready: JAX {version} on {devices}")
"""

STEP4 = r"""# Step 4: start Jupyter inside Ubuntu (on the GPU) and open WikiBandits
def wsl_path(p):
    m = re.match(r"^([A-Za-z]):[\\/]*(.*)$", str(p))
    if not m:
        raise Stop(f"This folder ({p}) isn't on a drive letter. Move the notebooks to a normal folder "
                   "(e.g. Documents) and Run all again.")
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/").rstrip("/")


def server_host():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for host in ("127.0.0.1", "localhost"):
        try:
            with opener.open(f"http://{host}:{PORT}/api", timeout=3) as r:
                if r.status == 200:
                    return host
        except Exception:
            pass
    return None


folder = wsl_path(HERE)
if "'" in folder:
    raise Stop("The folder path contains an apostrophe ('). Rename the folder or move the notebooks, then Run all again.")
try:
    saved = json.loads(STATE.read_text())
except (OSError, ValueError):
    saved = {}

host = server_host()
if host and saved.get("port") == PORT and saved.get("folder") == folder:
    token = saved["token"]
    print("The GPU Jupyter is already running.")
elif host:
    raise Stop(f"Something else is using port {PORT} (maybe a GPU Jupyter started from another folder). "
               "Run the stop cell at the bottom, or change PORT in the first cell, then run this cell again.")
else:
    token = secrets.token_hex(24)
    script = " && ".join([
        f"cd {shlex.quote(folder)}",
        "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75",
        f"exec {VENV}/bin/jupyter notebook --allow-root --no-browser --ip=127.0.0.1 --port={PORT} "
        f"--IdentityProvider.token={token} --ServerApp.root_dir={shlex.quote(folder)}"])
    log_path = HERE / "gpu_jupyter.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen(wsl_args(script), stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=ENV, creationflags=DETACHED)
    print("Starting Jupyter inside Ubuntu", end="", flush=True)
    for _ in range(60):
        time.sleep(2)
        print(".", end="", flush=True)
        host = server_host()
        if host or proc.poll() is not None:
            break
    print()
    if not host:
        tail = log_path.read_text(errors="replace").splitlines()[-15:]
        raise Stop("Jupyter didn't start inside Ubuntu. Last lines of gpu_jupyter.log:\n" + "\n".join(tail))
    STATE.write_text(json.dumps({"port": PORT, "token": token, "folder": folder}))

notebooks = sorted((p for p in HERE.glob("*.ipynb") if "wikibandits" in p.name.lower()),
                   key=lambda p: (p.name != "WikiBandits.ipynb", -p.stat().st_mtime))
base = f"http://{host}:{PORT}"
links = [(f"{base}/notebooks/{urllib.parse.quote(p.name)}?token={token}", p.name) for p in notebooks]
links.append((f"{base}/tree?token={token}", "the whole folder"))
display(HTML("<p><b>Open on the GPU:</b></p><ul>"
             + "".join(f'<li><a href="{u}" target="_blank">{n}</a></li>' for u, n in links) + "</ul>"))
if notebooks:
    webbrowser.open(links[0][0])
print("In the new tab: if Jupyter asks for a kernel, choose 'Python 3'. Section 1 of WikiBandits should say GPU.")
print("If WikiBandits.ipynb is also open in this Jupyter, close it here (File > Close and Shut Down Notebook).")
print("The GPU Jupyter keeps running if you close this tab; it stops when Windows restarts or with the cell below.")
"""

STOP_MD = r"""### Stopping the GPU Jupyter

Only needed if you want to free the GPU or restart it. **This also stops any training running in it**
(finished runs are saved; the one in progress restarts from its beginning next time).
"""

STOP = r"""STOP_GPU_JUPYTER = False   # set to True, then run this cell
if STOP_GPU_JUPYTER:
    run(["wsl.exe", "--terminate", DISTRO])
    STATE.unlink(missing_ok=True)
    print("Stopped. Run Step 4 again to start it.")
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(INTRO),
            nbf.v4.new_code_cell(SETTINGS),
            nbf.v4.new_code_cell(STEP1),
            nbf.v4.new_code_cell(STEP2),
            nbf.v4.new_code_cell(STEP3),
            nbf.v4.new_code_cell(STEP4),
            nbf.v4.new_markdown_cell(STOP_MD),
            nbf.v4.new_code_cell(STOP)]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "GPU_on_Windows.ipynb")
print("wrote GPU_on_Windows.ipynb")
