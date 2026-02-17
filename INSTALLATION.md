# EchoZero Installation Guide

## Overview

This guide covers proper installation of EchoZero on a new machine, with special attention to the Demucs dependency which requires careful Python environment management.

## Quick Reference (TL;DR)

**For a new machine installation:**

```bash
# 1. Clone repository
git clone https://github.com/yourusername/EchoZero.git
cd EchoZero

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment (IMPORTANT!)
source venv/bin/activate  # macOS/Linux
# OR: venv\Scripts\activate  # Windows

# 4. Install all dependencies (includes Demucs)
pip install -r requirements.txt

# 5. Install EchoZero
pip install -e .

# 6. Verify Demucs is installed
demucs --help

# 7. Run application (with venv activated!)
python main_qt.py
```

**Key Point:** Always activate the virtual environment before running EchoZero. The `demucs` command must be in your PATH, which happens automatically when the venv is activated.

## Prerequisites

- Python 3.10 or higher (3.10, 3.11, or 3.12)
- pip (Python package manager)
- FFmpeg (for audio format support - usually pre-installed on macOS/Linux)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/EchoZero.git
cd EchoZero
```

### 2. Create a Virtual Environment (Recommended)

**Why use a virtual environment?**
- Ensures all dependencies (including Demucs) are installed in the same Python environment
- Prevents conflicts with system Python packages
- Makes the installation reproducible

**Create and activate virtual environment:**

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Verify activation:**
- Your terminal prompt should show `(venv)` at the beginning
- Running `which python` (macOS/Linux) or `where python` (Windows) should point to the venv

### 3. Install Dependencies

**Install all dependencies (including Demucs):**
```bash
pip install -r requirements.txt
```

This installs:
- Demucs (>=4.0.0) - Audio source separation
- PyTorch and torchaudio - Machine learning frameworks
- All other EchoZero dependencies

**Install EchoZero in development mode:**
```bash
pip install -e .
```

### 4. Verify Demucs Installation

**Check that Demucs command is available:**
```bash
which demucs        # macOS/Linux
where demucs        # Windows
```

**Test Demucs installation:**
```bash
demucs --help
```

**Verify Demucs Python module:**
```bash
python -c "import demucs; print('Demucs installed successfully')"
```

**Expected output:**
- `demucs` command should be in your PATH (typically in `venv/bin/demucs`)
- `demucs --help` should show Demucs usage information
- Python import should succeed without errors

### 5. Launch EchoZero

**GUI:**
```bash
python main_qt.py
```

**CLI:**
```bash
python main.py
# or
echozero
```

## Troubleshooting

### Issue: "Demucs is not installed or not in PATH"

**Symptoms:**
- Error message: "Demucs is not installed or not in PATH"
- `which demucs` returns nothing
- `demucs --help` fails

**Causes:**
1. Demucs not installed in the active Python environment
2. Virtual environment not activated
3. PATH doesn't include the virtual environment's bin directory

**Solutions:**

**1. Ensure virtual environment is activated:**
```bash
# Check if venv is active (should show venv path)
which python

# If not active, activate it:
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

**2. Install Demucs in the active environment:**
```bash
pip install demucs>=4.0.0
```

**3. Verify installation:**
```bash
which demucs
demucs --help
```

**4. If still not found, check Python environment:**
```bash
# Check which Python is being used
python --version
which python

# Install directly with that Python
python -m pip install demucs>=4.0.0

# Verify the command location
python -m demucs --help
```

### Issue: Demucs installed but command not found

**Symptoms:**
- `python -c "import demucs"` works
- `demucs --help` fails
- `which demucs` returns nothing

**Cause:** The `demucs` command-line script wasn't installed or isn't in PATH.

**Solution:**
```bash
# Reinstall Demucs to ensure CLI script is installed
pip uninstall demucs
pip install demucs>=4.0.0

# Check if demucs script exists in venv/bin
ls venv/bin/demucs  # macOS/Linux
dir venv\Scripts\demucs.exe  # Windows

# If script exists but not in PATH, add venv/bin to PATH
export PATH="$(pwd)/venv/bin:$PATH"  # macOS/Linux (temporary)
```

### Issue: Multiple Python versions causing conflicts

**Symptoms:**
- Demucs installed in one Python version
- Application runs with different Python version
- `which demucs` points to different Python than `which python`

**Solution:**
1. **Always use virtual environment** (see step 2 above)
2. **Activate venv before running application**
3. **Use the same Python for everything:**
   ```bash
   # Check Python version
   python --version
   
   # Use this Python to create venv
   python -m venv venv
   
   # Activate and install
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Issue: SSL Certificate Errors (macOS)

**Symptoms:**
- Demucs fails to download models
- SSL certificate verification errors

**Solution:**
```bash
# macOS: Install certificates
/Applications/Python\ 3.*/Install\ Certificates.command

# Or set SSL certificate path
export SSL_CERT_FILE=$(python -m certifi)
```

### Issue: Models not downloading

**Symptoms:**
- Demucs runs but fails to find models
- First-time use errors

**Solution:**
- Models download automatically on first use
- Ensure internet connection
- Check disk space (~1 GB needed for models)
- Models are cached in `~/.cache/torch/hub/checkpoints/`

## Installation Verification Checklist

After installation, verify:

- [ ] Virtual environment is activated (`(venv)` in prompt)
- [ ] `python --version` shows 3.10+
- [ ] `pip list` shows `demucs` installed
- [ ] `which demucs` points to venv location
- [ ] `demucs --help` works
- [ ] `python -c "import demucs"` succeeds
- [ ] `python main_qt.py` launches GUI without errors
- [ ] Separator block can be added and configured

## Current Machine Analysis

**On this machine:**
- Demucs is installed in Python 3.12: `/Library/Frameworks/Python.framework/Versions/3.12/bin/demucs`
- Application runs with Python 3.10: `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3`
- **Problem:** Mismatch between Python versions

**To fix on this machine:**
```bash
# Option 1: Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Option 2: Install Demucs in Python 3.10
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m pip install demucs>=4.0.0
```

## Best Practices for New Machines

1. **Always use a virtual environment** - Prevents Python version conflicts
2. **Activate venv before running** - Ensures correct Python environment
3. **Install from requirements.txt** - Ensures version compatibility
4. **Verify installation** - Run verification steps before first use
5. **Keep venv activated** - When running EchoZero, always have venv active

## Additional Notes

- **Demucs version:** Pinned to >=4.0.0 in requirements.txt for stability
- **Model downloads:** Models (~1 GB) download automatically on first use
- **GPU support:** Demucs automatically uses GPU if available (CUDA/MPS)
- **Performance:** GPU provides 30-50x speedup over CPU
- **Apple Silicon:** Currently CPU-only (MPS support pending PyTorch FFT operations)

## Getting Help

If installation issues persist:
1. Check Python version: `python --version`
2. Check virtual environment: `which python`
3. Check Demucs installation: `pip list | grep demucs`
4. Check Demucs command: `which demucs`
5. Review error messages carefully
6. Ensure all prerequisites are met
