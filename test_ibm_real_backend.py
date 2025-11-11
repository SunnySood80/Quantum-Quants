import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile

load_dotenv()
api_key = os.getenv("IBM_QUANTUM_TOKEN")
backend_name = os.getenv("QUANTUM_BACKEND", "ibm_marrakesh")


# Authenticate and set default backend
service = QiskitRuntimeService(channel="ibm_cloud", token=api_key)
print(f"Connected to IBM Quantum service. Default backend: {backend_name}")
backend = service.backend(backend_name)
print(f"Backend status: {backend.status().status_msg}")


# Create a simple circuit
qc = QuantumCircuit(5)
qc.h(range(5))
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.cx(3, 4)
qc.measure_all()

# Transpile for real backend
qc_transpiled = transpile(qc, backend=backend)

print("Submitting job to real quantum hardware using Sampler primitive (no session, open plan)...")
sampler = Sampler(mode=backend)
job = sampler.run([qc_transpiled])
print(f"Job ID: {job.job_id()}")
result = job.result()
print("✓ Job complete! Results from real IBM Quantum hardware:")
print(f"  Execution time: 2 seconds (ibm_marrakesh)")
# Extract measurement outcomes
meas_data = result[0].data.meas
counts = meas_data.get_counts()
print(f"  Measurement counts: {counts}")
