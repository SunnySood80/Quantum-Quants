# 🎛️ Quantum Hardware Toggle Guide

## Quick Start: One Variable to Rule Them All!

To switch between simulator and real quantum hardware, just change **ONE variable** in `main_portfolio_optimization.py`:

```python
# Line ~622 in main_portfolio_optimization.py
USE_QUANTUM_HARDWARE = False  # ← Change this to True for real quantum computer!
BACKEND_NAME = "ibm_brisbane"  # ← Choose your quantum backend
```

That's it! Everything else is handled automatically. 🚀

---

## Available Quantum Backends

Choose from IBM's quantum computers (127 qubits each):

- `ibm_brisbane` - Brisbane, Australia
- `ibm_kyoto` - Kyoto, Japan  
- `ibm_sherbrooke` - Sherbrooke, Canada
- `ibm_osaka` - Osaka, Japan

Or let the system auto-select:
```python
backend_name = "least_busy"  # Automatically picks the least busy quantum computer
```

---

## First-Time Setup (Real Hardware Only)

### Step 1: Install IBM Runtime
```bash
pip install qiskit-ibm-runtime
```

### Step 2: Create IBM Quantum Account
1. Go to: https://quantum.ibm.com/
2. Sign up for a free account
3. Navigate to your account settings
4. Copy your API token

### Step 3: Save Your Token (One-Time)
```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Run this once to save your credentials
QiskitRuntimeService.save_account(
    channel="ibm_quantum", 
    token="YOUR_API_TOKEN_HERE"
)
```

---

## Usage Examples

### Example 1: Run on Simulator (Default)
```python
# In main_portfolio_optimization.py (around line 622)
USE_QUANTUM_HARDWARE = False
BACKEND_NAME = "ibm_brisbane"  # Not used in simulator mode

# Then just run:
python main_portfolio_optimization.py
```

Output:
```
💻 Using local quantum simulator...
```

### Example 2: Run on Real Quantum Hardware
```python
# In main_portfolio_optimization.py (around line 622)
USE_QUANTUM_HARDWARE = True
BACKEND_NAME = "ibm_brisbane"

# Then just run:
python main_portfolio_optimization.py
```

Output:
```
🔌 Connecting to IBM Quantum Cloud...
✓ Connected to: ibm_brisbane
  - Backend qubits: 127
  - Status: active
  - Pending jobs: 5
```

---

## What Gets Adjusted Automatically

When you toggle `USE_QUANTUM_HARDWARE`, the system automatically adjusts:

| Parameter | Simulator Mode | Real Hardware Mode | Why? |
|-----------|---------------|-------------------|------|
| `qaoa_depth` | 3 | 2 | Shallower circuits = fewer errors on real hardware |
| `maxiter` | 300 | 5 | Real HW is slower, fewer iterations needed |
| `noise_level` | 0.01 | N/A | Real hardware has its own noise |
| Backend | `AerSimulator` | IBM Quantum | Different execution engines |
| Shots | Default | 1024 | Number of measurements per circuit |

---

## Parameter Recommendations

### For Development/Testing (Simulator):
```python
USE_QUANTUM_HARDWARE = False
latent_dim = 16          # 16 qubits
qaoa_depth = 3           # ~5-7 min runtime
maxiter = 300            # Good convergence
```

### For Production (Real Hardware):
```python
USE_QUANTUM_HARDWARE = True
BACKEND_NAME = "ibm_brisbane"
latent_dim = 16          # 16 qubits (hardware ready!)
qaoa_depth = 2           # Fewer errors
maxiter = 5              # Fast execution (~2 min)
```

---

## Expected Runtime

| Mode | Data Load | Autoencoder | QAOA | Total |
|------|-----------|-------------|------|-------|
| **Simulator** (first run) | 1-2 min | 3-4 min | 5-7 min | ~10-13 min |
| **Simulator** (cached) | 5 sec | 3-4 min | 5-7 min | ~8-11 min |
| **Real Hardware** (first run) | 1-2 min | 3-4 min | 2-3 min | ~6-9 min |
| **Real Hardware** (cached) | 5 sec | 3-4 min | 2-3 min | ~3-7 min |

---

## Troubleshooting

### Error: "qiskit-ibm-runtime not installed"
**Solution:** 
```bash
pip install qiskit-ibm-runtime
```

### Error: "No IBM Quantum credentials found"
**Solution:** Save your API token:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```

### Error: "Backend not available"
**Solution:** Check backend status at https://quantum.ibm.com/services/resources or use:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum")
print(service.backends())  # List all available backends
```

### Long Queue Times
**Solution:** Choose a less busy backend or use auto-select:
```python
BACKEND_NAME = "least_busy"
```

---

## Code Locations

| What | File | Line(s) |
|------|------|---------|
| **Main Toggle** | `main_portfolio_optimization.py` | ~622 |
| Backend Setup | `qubo_portfolio_optimizer.py` | 213-275 |
| Parameter Auto-Adjust | `main_portfolio_optimization.py` | 650-659 |

---

## Benefits of This Design

✅ **One variable to switch** - No commenting/uncommenting code blocks  
✅ **Auto-adjusts parameters** - Optimal settings for each mode  
✅ **Graceful error handling** - Clear messages if setup incomplete  
✅ **Backward compatible** - Defaults to simulator if real HW unavailable  
✅ **Easy debugging** - Can quickly test on simulator first  

---

## Advanced: Direct Backend Selection

If you want more control, you can directly configure the backend in your code:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Get service
service = QiskitRuntimeService(channel="ibm_quantum")

# List all backends
backends = service.backends()
for backend in backends:
    print(f"{backend.name}: {backend.num_qubits} qubits, {backend.status().pending_jobs} jobs pending")

# Auto-select least busy with 16+ qubits
backend = service.least_busy(
    operational=True,
    simulator=False, 
    min_num_qubits=16
)
print(f"Selected: {backend.name}")
```

---

## Questions?

- IBM Quantum Docs: https://docs.quantum.ibm.com/
- Qiskit Runtime: https://qiskit.org/ecosystem/ibm-runtime/
- Get Help: https://quantumcomputing.stackexchange.com/

Happy quantum computing! 🚀✨

