"""
IBM Quantum Credentials Setup
PASTE YOUR TOKEN BELOW AND RUN THIS SCRIPT ONCE
"""

# ============================================================================
# 🔐 PASTE YOUR IBM QUANTUM TOKEN HERE (between the quotes):
# ============================================================================
TOKEN = "5dAkOSJu4eanjVdNiZT5pDfkHZutxZeukIh8SkAYONEL"  # ← PUT YOUR TOKEN HERE
# ============================================================================

print("=" * 70)
print(" " * 15 + "IBM QUANTUM CREDENTIALS SETUP")
print("=" * 70)
print()
print("Validating token...")

token = TOKEN.strip()

if not token:
    print()
    print("❌ No token provided. Please paste your token in the code (line 9).")
    exit(1)

print()
print("Saving credentials...")
print("-" * 70)

try:
    # Try to import and save
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    # Save the account - MUST USE "ibm_cloud" channel
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=token,
        overwrite=True  # Allow updating if token already exists
    )
    
    print("✅ Success! Your credentials have been saved.")
    print()
    print("Testing connection...")
    print("-" * 70)
    
    # Test the connection
    service = QiskitRuntimeService(channel="ibm_cloud")
    backends = service.backends()
    
    print(f"✅ Connected successfully!")
    print(f"✅ Found {len(backends)} available quantum computers:")
    print()
    
    # Show available backends
    print("Available Quantum Computers:")
    print("-" * 70)
    for backend in backends[:10]:  # Show first 10
        status = backend.status()
        num_qubits = backend.num_qubits
        is_simulator = backend.configuration().simulator
        
        if not is_simulator and num_qubits >= 16:
            print(f"  ✓ {backend.name:<25} {num_qubits:>3} qubits    ({status.pending_jobs} jobs pending)")
    
    print()
    print("=" * 70)
    print("🎉 SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("You're ready to run on real quantum hardware!")
    print()
    print("Next step:")
    print("  👉 Run: python main_portfolio_optimization.py")
    print()
    print("=" * 70)

except ImportError:
    print("❌ ERROR: qiskit-ibm-runtime is not installed")
    print()
    print("Please install it first:")
    print("  👉 pip install qiskit-ibm-runtime")
    print()
    print("Then run this script again.")
    exit(1)

except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("Common issues:")
    print("  - Invalid token (check you copied it correctly)")
    print("  - Network connection issues")
    print("  - IBM Quantum services temporarily unavailable")
    print()
    print("Please try again or visit: https://docs.quantum.ibm.com/")
    exit(1)

