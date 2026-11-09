# GTP Engine Setup

Place engine binaries under `gtp_agents/` so the registry auto-discovers them.
The directory layout expected by `find_binary()` is shown for each engine.

---

## GNU Go (classical, weak baseline)

**WSL / Linux:**
```bash
sudo apt install gnugo          # already on PATH at /usr/games/gnugo
```

**Windows (pre-built):**
1. Download `gnugo-3.8-win32.exe` from https://gnugo.sourceforge.net/
2. Extract into `gtp_agents/` so you have:
   ```
   gtp_agents/gnugo-3.8/gnugo-3.8/gnugo.exe
   ```

**Registry usage:** `gnugo` or `gnugo:level=5`

---

## Pachi (MCTS, configurable strength)

**WSL / Linux:**
```bash
# Build from source (no apt package)
cd /tmp
git clone https://github.com/pasky/pachi.git
cd pachi
make
cp pachi ~/bin/   # or anywhere on PATH
```

**Windows (pre-built):**
1. Download from https://github.com/pasky/pachi/releases (look for `pachi-*-win64-avx.zip`)
2. Extract into `gtp_agents/` so you have:
   ```
   gtp_agents/pachi-12.88-win64-avx/Pachi-12.88/pachi.exe
   ```

**Registry usage:** `pachi` or `pachi:playouts=5000,threads=4`

> Note: Pachi requires >= 500 playouts. The registry enforces this automatically.

---

## Fuego (classical MCTS, different style from GNU Go)

**WSL / Linux:**
```bash
sudo apt install fuego
# Installs to /usr/games/fuego
```

**Windows (build from source):**
```bash
# Fuego has no pre-built Windows binary. Options:
# 1. Use WSL (recommended)
# 2. Build with CMake + Visual Studio:
git clone https://github.com/gouct/fuego.git
cd fuego
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
# Copy fuego.exe to gtp_agents/fuego/fuego.exe
```

Place the binary so you have:
```
gtp_agents/fuego/fuego.exe      # Windows
gtp_agents/fuego/fuego           # Linux
```

**Registry usage:** `fuego` or `fuego:max_games=5000`

---

## KataGo (state-of-the-art, adjustable strength)

KataGo needs both a binary and a neural network weights file.

**WSL / Linux:**
```bash
# Option 1: Pre-built binary
# Download from https://github.com/lightvector/KataGo/releases
# Get the OpenCL or CUDA version matching your GPU setup
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip
unzip katago-v1.15.3-opencl-linux-x64.zip -d gtp_agents/katago/

# Option 2: apt (may be outdated)
sudo apt install katago
```

**Windows:**
```bash
# Download from https://github.com/lightvector/KataGo/releases
# Get katago-v1.15.3-opencl-windows-x64.zip (or cuda if you have NVIDIA GPU)
# Extract so you have:
#   gtp_agents/katago/katago.exe
```

**Weights file (required for both platforms):**
```bash
# Download a network from https://katagotraining.org/networks/
# Recommended: the latest "b18c384" or smaller for faster eval
# Place it anywhere, pass the path to the registry:
wget -O gtp_agents/katago/model.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
```

You also need a GTP config file. Generate a default one:
```bash
gtp_agents/katago/katago genconfig -model gtp_agents/katago/model.bin.gz -output gtp_agents/katago/gtp.cfg
```
Or use a minimal config (the registry passes `maxVisits` via override anyway).

**Registry usage:**
```
katago:model=gtp_agents/katago/model.bin.gz,max_visits=100
katago:model=gtp_agents/katago/model.bin.gz,max_visits=1,config=gtp_agents/katago/gtp.cfg
```

> Tip: `max_visits=1` makes KataGo play its raw neural network policy (no search),
> which is a fair comparison against your pure policy networks.
> `max_visits=10` adds light search. `max_visits=1000` is near full strength.

---

## Leela (original, neural network engine)

**WSL / Linux:**
```bash
# Build from source
git clone https://github.com/gcp/Leela.git
cd Leela
cmake . -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Copy leela binary to gtp_agents/leela/leela
```

**Windows:**
1. Download from https://www.sjeng.org/leela.html (older releases have Windows builds)
2. Extract so you have:
   ```
   gtp_agents/leela/leela.exe
   ```

**Registry usage:** `leela` or `leela:playouts=1000`

> Note: Leela (original) is much weaker than Leela Zero / KataGo.
> It's useful as a mid-range opponent between GNU Go and KataGo.

---

## Quick verification

After setup, verify each engine responds to GTP:
```bash
echo -e "name\nversion\nquit" | gtp_agents/gnugo-3.8/gnugo-3.8/gnugo.exe --mode gtp
echo -e "name\nversion\nquit" | gtp_agents/fuego/fuego
echo -e "name\nversion\nquit" | gtp_agents/katago/katago gtp -model gtp_agents/katago/model.bin.gz
```

Or use the registry's built-in test:
```bash
python -c "from alphigo.agent.registry import create_agent; a = create_agent('gnugo'); print('OK')"
python -c "from alphigo.agent.registry import create_agent; a = create_agent('fuego'); print('OK')"
```
