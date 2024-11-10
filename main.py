# General
import numpy as np

# Plotting routines
import matplotlib.pyplot as plt
from collections import defaultdict

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Runtime imports
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, SamplerOptions
from qiskit.providers.jobstatus import JobStatus  # Corrected import

# Function that creates a maximally entangled Bell state
def BellCircuit(qreg_name='q', creg_name='c') -> QuantumCircuit:
    qreg_q = QuantumRegister(2, qreg_name)
    creg_c = ClassicalRegister(2, creg_name)
    qc = QuantumCircuit(qreg_q, creg_c)
    qc.h(qreg_q[0])
    qc.cx(qreg_q[0], qreg_q[1])
    qc.barrier()
    return qc

bell = BellCircuit()
bell.draw(output='mpl')
plt.show()

# Define the views
views = ("AB", "Ab", "aB", "ab")

def MeasureWithView(circuit: QuantumCircuit, view: str):
    if view == "AB":
        circuit.ry(-np.pi/4, circuit.qubits[1])
        circuit.measure([0, 1], [0, 1])
    elif view == "Ab":
        circuit.ry(np.pi/4, circuit.qubits[1])
        circuit.measure([0, 1], [0, 1])
    elif view == "aB":
        circuit.ry(-np.pi/4, circuit.qubits[1])
        circuit.h(circuit.qubits[0])
        circuit.measure([0, 1], [0, 1])
    elif view == "ab":
        circuit.ry(np.pi/4, circuit.qubits[1])
        circuit.h(circuit.qubits[0])
        circuit.measure([0, 1], [0, 1])
    return circuit

# Create circuits for each view
circuits = [MeasureWithView(BellCircuit('q' + view, 'c' + view), view) for view in views]

# Plot the circuits
for circuit, view in zip(circuits, views):
    print(f"Circuit for view {view}:")
    circuit.draw(output='mpl')
    plt.show()

# Function to plot multiple circuits
def PlotCircuits(*circuits: QuantumCircuit):
    num_circuits = len(circuits)
    fig, axs = plt.subplots(1, num_circuits, figsize=(4 * num_circuits, 4))
    if num_circuits == 1:
        axs = [axs]
    for i, (circuit, ax) in enumerate(zip(circuits, axs)):
        circuit.draw(output='mpl', ax=ax)
        ax.set_title(f'Circuit {i+1}')
    plt.tight_layout()
    plt.show()

PlotCircuits(*circuits)

# Function to plot circuit results
def PlotCircuitResults(circuit_results: dict):
    try:
        formatted_results = {}
        for view, counts in circuit_results.items():
            if isinstance(counts, dict):
                # Ensure keys are strings and values are numbers
                formatted_counts = {}
                for k, v in counts.items():
                    if not isinstance(k, str):
                        k = str(k)
                    if isinstance(v, dict):
                        print(f"Nested dict found in counts for key {k} in view {view}: {v}")
                        # Handle or skip nested dicts
                        continue
                    if not isinstance(v, (int, float)):
                        print(f"Invalid count value for key {k} in view {view}: {v}")
                        continue
                    formatted_counts[k] = v
                formatted_results[view] = formatted_counts
            else:
                print(f"Counts for view {view} are not in the expected format.")
        plot_histogram(formatted_results, figsize=(8,6))
        plt.show()
    except Exception as e:
        print(f"Error encountered while plotting results: {e}")

# Example usage of PlotCircuitResults
PlotCircuitResults({
    "AB": {"00": 400, "01": 300, "10": 200, "11": 100},
    "Ab": {"00": 100, "01": 200, "10": 300, "11": 400},
    "aB": {"00": 250, "01": 250, "10": 250, "11": 250},
    "ab": {"00": 900, "01": 0, "10": 0, "11": 100},
})

# Get the Qiskit Runtime service
service = QiskitRuntimeService()

# Select the backend
backend_name = 'ibm_kyiv'  # Use 'ibm_kyiv', 'ibm_brisbane', or 'ibm_sherbrooke'
backend = service.backend(backend_name)

# Print backend information
num_qubits = backend.configuration().n_qubits
print(f"Using backend: {backend_name} with {num_qubits} qubits.")

# Create a combined circuit that runs all 4 experiments simultaneously
circ_combined = QuantumCircuit()

# In a similar manner, create an even wider circuit using copies of the combined experiments circuit
# Utilize roughly 80% of the machine to accommodate broken connections between qubits and topology
num_qubits = 8 * ((backend.num_qubits // 5 * 4) // 8)

circ_real = QuantumCircuit()

for i in range(0, num_qubits, 8):
    for j, view in enumerate(views):
        # Create unique register names for each experiment copy
        qreg_name = f'q{i + j * 2}'
        creg_name = f'c{i + j * 2}'

        # Create a fresh Bell circuit with unique register names
        experiment_circuit = MeasureWithView(BellCircuit(qreg_name, creg_name), view)

        # Add the registers to the real circuit
        circ_real.add_register(experiment_circuit.qregs[0])
        circ_real.add_register(experiment_circuit.cregs[0])

        # Determine the qubit and clbit indices in circ_real where to compose the experiment
        qubit_indices = list(range(i + j * 2, i + j * 2 + 2))  # 2 qubits per experiment
        clbit_indices = list(range(i + j * 2, i + j * 2 + 2))  # 2 classical bits per experiment

        # Compose the experiment circuit into circ_real at the specified indices
        circ_real.compose(
            experiment_circuit,
            qubits=qubit_indices,
            clbits=clbit_indices,
            inplace=True
        )

# Optional: Draw the real circuit to verify (commented out to avoid large plots)
# circ_real.draw('mpl')
# plt.show()

# Adjust num_repeats based on the backend's qubit count
circuit_qubits = circ_combined.num_qubits
num_repeats = num_qubits // circuit_qubits

if num_repeats == 0:
    num_repeats = 1  # Ensure at least one repetition

print(f"Number of repeats: {num_repeats}")

# Build the real circuit by repeating the combined circuit
circ_real = QuantumCircuit()
for repeat in range(num_repeats):
    offset = repeat * circuit_qubits
    # Shift qubit and clbit indices for each repetition
    circ_real.add_register(QuantumRegister(circuit_qubits, name=f'q{repeat}'))
    circ_real.add_register(ClassicalRegister(circuit_qubits, name=f'c{repeat}'))
    circ_real.compose(
        circ_combined,
        qubits=range(offset, offset + circuit_qubits),
        clbits=range(offset, offset + circuit_qubits),
        inplace=True
    )

# Comment out drawing large circuit to avoid matplotlib error
# circ_real.draw('mpl')
# plt.show()

# Transpile the combined circuit for the real backend
pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)  # Corrected function call
transpiled_circ_real = pass_manager.run(circ_real)
print("Depth of transpiled circuit:", transpiled_circ_real.depth())

# Run the transpiled circuit on the real backend using the Sampler
sampler_options = SamplerOptions(resilience_level=1)
sampler = Sampler(backend=backend, options=sampler_options)

job = sampler.run([transpiled_circ_real])
job_id = job.job_id()
print(f"Job ID: {job_id}")

# Monitor job status
import time
while True:
    status = job.status()
    print(f"Job status: {status}")
    if status == JobStatus.DONE or status == JobStatus.ERROR:
        break
    time.sleep(10)

# After the job is completed
result = job.result()

# Initialize real_counts dictionary
real_counts = {view: defaultdict(float) for view in views}

# Process the quasi-distribution
quasi_dist = result.quasi_dists[0]

# Total number of shots
total_shots = 1000  # Adjust this if necessary

# Convert quasi probabilities to probabilities
probs = quasi_dist.nearest_probability_distribution()

# Convert to counts
counts = {int(k): v * total_shots for k, v in probs.items()}

# Format the keys to bitstrings
num_clbits = circ_real.num_clbits
counts = {format(k, f'0{num_clbits}b'): v for k, v in counts.items()}

# Process counts for each repetition and view
for bitstring, value in counts.items():
    for repeat in range(num_repeats):
        offset = repeat * circ_combined.num_clbits
        segment = bitstring[offset:offset + circ_combined.num_clbits]
        index = 0
        for view in views:
            view_bits = segment[index:index + 2]
            if not isinstance(value, (int, float)):
                print(f"Invalid count value for bitstring {bitstring}: {value}")
                continue
            real_counts[view][view_bits] += value
            index += 2

# Convert counts to regular dicts for plotting
real_counts = {view: dict(count_dict) for view, count_dict in real_counts.items()}

# Print real_counts to inspect the data structure
print("real_counts:", real_counts)

# Plot the results
PlotCircuitResults(real_counts)

#ADD CODE

# EXTENSION HAS BEEN APPROVED FOR NOVEMBER 10th.