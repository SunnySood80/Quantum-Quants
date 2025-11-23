# 🔐 IBM Quantum Credentials Setup (Simple 3-Step Guide)

## ⚡ Quick Setup (5 minutes)

### **Step 1: Install IBM Quantum Library**

Open your terminal and run:

```bash
pip install qiskit-ibm-runtime
```

Wait for it to finish (takes ~30 seconds).

---

### **Step 2: Get Your API Token**

1. **Open this link:** https://quantum.ibm.com/

2. **Sign up** (or log in if you have an account)
   - Use any email
   - Free account (no credit card needed)

3. **After logging in:**
   - Click your profile icon (top right corner)
   - Click "Account settings"
   - OR go directly to: https://quantum.ibm.com/account

4. **Copy your API token:**
   - Look for "API token" section
   - Click the "Copy" button
   - Your token looks like this: `a1b2c3d4e5f6...` (long string)

---

### **Step 3: Save Your Token**

Run the setup script I created for you:

```bash
python setup_ibm_quantum.py
```

**It will ask you to:**
1. Paste your API token
2. Press Enter

**That's it!** ✅

The script will:
- Save your credentials
- Test the connection
- Show you available quantum computers

---

## 🎯 Complete Example (What You'll See)

```bash
# Step 1: Install
C:\Users\flurr\quantum quant> pip install qiskit-ibm-runtime
Installing...
✓ Successfully installed

# Step 2: Run setup script
C:\Users\flurr\quantum quant> python setup_ibm_quantum.py

======================================================================
               IBM QUANTUM CREDENTIALS SETUP
======================================================================

Follow these simple steps:

STEP 1: Get Your API Token
----------------------------------------------------------------------
  1. Open this link in your browser:
     👉 https://quantum.ibm.com/
  
  2. Sign up (or log in if you already have an account)
  
  3. After logging in, click your profile icon (top right)
  
  4. Go to 'Account settings' or visit directly:
     👉 https://quantum.ibm.com/account
  
  5. Look for 'API token' section
  
  6. Click 'Copy token' button

======================================================================

STEP 2: Paste your API token here and press Enter:
>>> [PASTE YOUR TOKEN HERE]

STEP 3: Saving your credentials...
----------------------------------------------------------------------
✅ Success! Your credentials have been saved.

Testing connection...
----------------------------------------------------------------------
✅ Connected successfully!
✅ Found 15 available quantum computers:

Available Quantum Computers:
----------------------------------------------------------------------
  ✓ ibm_brisbane              127 qubits    (3 jobs pending)
  ✓ ibm_kyoto                 127 qubits    (5 jobs pending)
  ✓ ibm_sherbrooke            127 qubits    (2 jobs pending)
  ✓ ibm_osaka                 127 qubits    (4 jobs pending)

======================================================================
🎉 SETUP COMPLETE!
======================================================================

You're ready to run on real quantum hardware!

Next step:
  👉 Run: python main_portfolio_optimization.py

======================================================================
```

---

## 🚀 After Setup - Run Your Code

Once credentials are saved, just run your main script:

```bash
python main_portfolio_optimization.py
```

It will automatically:
1. Connect to IBM Quantum
2. Select the quantum computer (ibm_brisbane)
3. Run your portfolio optimization
4. Return results

---

## ❓ Troubleshooting

### "qiskit-ibm-runtime not installed"
**Fix:**
```bash
pip install qiskit-ibm-runtime
```

### "Invalid token"
**Fix:**
1. Go back to https://quantum.ibm.com/account
2. Copy the token again (carefully)
3. Run `python setup_ibm_quantum.py` again

### "Already saved, want to overwrite?"
**Fix:**
- The script automatically overwrites (flag: `overwrite=True`)
- Just paste your new token

### Want to manually save credentials?
**Alternative method:**
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_TOKEN_HERE"
)
```

---

## 📍 Where Are Credentials Stored?

After running the setup, your token is saved in:
- **Windows:** `C:\Users\YOUR_USERNAME\.qiskit\qiskit-ibm.json`
- **Mac/Linux:** `~/.qiskit/qiskit-ibm.json`

You only need to do this **ONCE**. The credentials persist forever.

---

## ✅ Ready to Run?

After setup completes successfully:

```bash
python main_portfolio_optimization.py
```

Look for this output:
```
🔌 Connecting to IBM Quantum Cloud...
✓ Connected to: ibm_brisbane
  - Backend qubits: 127
  - Status: active
  - Pending jobs: 3
```

If you see that, you're running on a **REAL quantum computer**! 🎉

---

## 🔄 Need Help?

1. Check: https://docs.quantum.ibm.com/
2. Re-run: `python setup_ibm_quantum.py`
3. Check token at: https://quantum.ibm.com/account

