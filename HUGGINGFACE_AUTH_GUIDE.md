# Hugging Face Authentication Guide

## 🔑 Why Authentication?

Some Hugging Face datasets (including the Smithsonian hackathon datasets) may require authentication to access.

---

## ✅ **Method 1: CLI Login (Recommended - One-Time Setup)**

This is the **easiest** method. Run once and you're done!

### Steps:

```powershell
# 1. Get your Hugging Face token
# Visit: https://huggingface.co/settings/tokens
# Click "New token" → Create a "Read" token

# 2. Login via CLI (one-time)
huggingface-cli login

# 3. Paste your token when prompted
# Token will be saved to: C:\Users\<username>\.huggingface\token
```

### Verify:

```powershell
# Check if you're logged in
huggingface-cli whoami
```

**Done!** All Python scripts will now automatically use this token.

---

## ✅ **Method 2: Environment Variable**

Good for automation or CI/CD pipelines.

### Windows PowerShell:

```powershell
# Temporary (current session only)
$env:HF_TOKEN = "hf_your_token_here"

# Permanent (user-level)
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_your_token_here', 'User')

# Verify
echo $env:HF_TOKEN
```

### In Python:

```python
import os
from huggingface_hub import login

# Automatically uses HF_TOKEN environment variable
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
```

---

## ✅ **Method 3: Programmatic Login**

Login directly in your Python script.

### Interactive (prompts for token):

```python
from huggingface_hub import login

# Will prompt for token
login()
```

### With Token String:

```python
from huggingface_hub import login

# Pass token directly (NOT recommended for committed code)
login(token="hf_your_token_here")
```

### From File:

```python
from huggingface_hub import login
from pathlib import Path

# Read token from a file
token_file = Path("my_token.txt")
token = token_file.read_text().strip()
login(token=token)
```

---

## 🎯 **What Our Script Does**

The `download_newspapers_simple.py` script automatically:

1. ✅ Checks if you're already logged in (token file exists)
2. ✅ Tries `HF_TOKEN` environment variable
3. ✅ Prompts for interactive login if needed
4. ✅ Continues without auth if you skip (public datasets still work)

### Script Behavior:

```python
def authenticate_huggingface():
    """Smart authentication with fallbacks."""
    # 1. Check existing token file
    if Path.home() / ".huggingface" / "token" exists:
        return True  # Already logged in
    
    # 2. Try environment variable
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
        return True
    
    # 3. Interactive prompt
    login()  # User enters token
    return True
```

---

## 🔐 **Security Best Practices**

### ✅ DO:
- Use `huggingface-cli login` for local development
- Use environment variables for automation
- Use "Read" tokens (not "Write") for downloading data
- Store tokens in `.env` files (add to `.gitignore`)

### ❌ DON'T:
- Hardcode tokens in Python files
- Commit tokens to Git repositories
- Share tokens in chat/email
- Use "Write" tokens unless necessary

---

## 📝 **Quick Reference**

| Method | Command | When to Use |
|--------|---------|-------------|
| **CLI Login** | `huggingface-cli login` | Local development (recommended) |
| **Environment Variable** | `$env:HF_TOKEN = "..."` | Automation, scripts |
| **Programmatic** | `login(token=...)` | Custom workflows |

---

## 🚀 **Quick Start for Hackathon**

```powershell
# 1. Get token from: https://huggingface.co/settings/tokens

# 2. Login once
huggingface-cli login

# 3. Run your script - authentication is automatic!
python download_newspapers_simple.py --limit 100
```

---

## 🐛 **Troubleshooting**

### "Token not found" error:

```powershell
# Check if token file exists
Test-Path "$env:USERPROFILE\.huggingface\token"

# If False, run login again
huggingface-cli login
```

### "Permission denied" error:

```powershell
# Check token permissions
# Visit: https://huggingface.co/settings/tokens
# Make sure token has "Read" access
```

### Script doesn't prompt for login:

```python
# Force authentication in script
from huggingface_hub import login
login()  # Will always prompt
```

---

## 📚 **Additional Resources**

- **Hugging Face Tokens:** https://huggingface.co/settings/tokens
- **HF Hub Documentation:** https://huggingface.co/docs/huggingface_hub
- **CLI Reference:** https://huggingface.co/docs/huggingface_hub/guides/cli

---

**Created:** May 7, 2026  
**For:** Smithsonian Hackathon - Track 2, Team 5
