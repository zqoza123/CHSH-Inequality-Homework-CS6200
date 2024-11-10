# General
import numpy as np

# Plotting routines
import matplotlib.pyplot as plt
from collections import defaultdict

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Aer imports
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Qiskit Runtime imports
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, SamplerOptions
from qiskit.providers.jobstatus import JobStatus

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
        circuit.ry(-np.pi / 4, circuit.qubits[1])
        circuit.measure([0, 1], [0, 1])
    elif view == "Ab":
        circuit.ry(np.pi / 4, circuit.qubits[1])
        circuit.measure([0, 1], [0, 1])
    elif view == "aB":
        circuit.ry(-np.pi / 4, circuit.qubits[1])
        circuit.h(circuit.qubits[0])
        circuit.measure([0, 1], [0, 1])
    elif view == "ab":
        circuit.ry(np.pi / 4, circuit.qubits[1])
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
        ax.set_title(f'Circuit {i + 1}')
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
                    if isinstance(v, dict):
                        print(f"Skipping nested dict for key {k} in view {view}.")
                        continue  # Skip nested dicts
                    if not isinstance(k, str):
                        k = str(k)
                    if not isinstance(v, (int, float)):
                        print(f"Invalid count value for key {k} in view {view}: {v}")
                        continue
                    formatted_counts[k] = v
                if formatted_counts:  # Only add if not empty
                    formatted_results[view] = formatted_counts
            else:
                print(f"Counts for view {view} are not in the expected format.")
        plot_histogram(formatted_results, figsize=(8, 6))
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


# --- Running Circuits on an Ideal Simulator ---

# Use AerSimulator to simulate 1000 shots for each view circuit
simulator = AerSimulator()
job = simulator.run(circuits, shots=1000)
ideal_results = job.result()

# Get counts for each circuit
ideal_counts = {view: ideal_results.get_counts(circuit) for view, circuit in zip(views, circuits)}

# Plot the results
PlotCircuitResults(ideal_counts)


# --- Running Circuits on an IBM Noisy Simulator ---

# Load saved credentials
service = QiskitRuntimeService()

# Get a backend
backend_name = 'ibm_kyiv'  # or another backend
backend = service.backend(backend_name)

# Create a NoiseModel using the backend
noise_model = NoiseModel.from_backend(backend)

# Create an AerSimulator using the noise model
noisy_simulator = AerSimulator(noise_model=noise_model)

# Transpile circuits for the noisy backend
transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)
transpiled_circuits = transpiler.run(circuits)

# Run the transpiled circuits on the noisy simulator
job = noisy_simulator.run(transpiled_circuits, shots=1000)
noisy_results = job.result()

# Get counts for each circuit
noisy_counts = {view: noisy_results.get_counts(circuit) for view, circuit in zip(views, transpiled_circuits)}

# Plot the results
PlotCircuitResults(noisy_counts)


# --- Running Circuits on an IBM Quantum Computer ---

# Define the combined circuit that includes all four view experiments
circ_combined = QuantumCircuit()

for view in views:
    # Create unique register names for each experiment within the combined circuit
    qreg_name = f'q_{view}'
    creg_name = f'c_{view}'

    # Create a fresh Bell circuit with unique register names
    experiment_circuit = MeasureWithView(BellCircuit(qreg_name, creg_name), view)

    # Add the registers to the combined circuit
    circ_combined.add_register(experiment_circuit.qregs[0])
    circ_combined.add_register(experiment_circuit.cregs[0])

    # Determine the qubit and clbit indices within circ_combined
    # Since circ_combined is being built sequentially, qubits and clbits are added in order
    qubit_indices = list(range(circ_combined.num_qubits - 2, circ_combined.num_qubits))
    clbit_indices = list(range(circ_combined.num_clbits - 2, circ_combined.num_clbits))

    # Compose the experiment circuit into circ_combined at the specified indices
    circ_combined.compose(
        experiment_circuit,
        qubits=qubit_indices,
        clbits=clbit_indices,
        inplace=True
    )

# Optional: Draw the combined circuit to verify
circ_combined.draw('mpl')
plt.show()

# Calculate the number of qubits to use (80% of total, rounded down to nearest multiple of 8)
num_qubits = 8 * ((backend.num_qubits * 4 // 5) // 8)
print(f"Total qubits in circ_real: {num_qubits}")

# Determine the number of repeats
circuit_qubits = circ_combined.num_qubits  # Should be 8 (2 qubits * 4 experiments)
num_repeats = num_qubits // circuit_qubits
print(f"Number of repeats: {num_repeats}")

# Build the real circuit by repeating the combined circuit
circ_real = QuantumCircuit()

for repeat in range(num_repeats):
    # Create unique register names for each repetition
    qreg_name = f'q_rep{repeat}'
    creg_name = f'c_rep{repeat}'

    # Create quantum and classical registers for this repetition
    qreg = QuantumRegister(circuit_qubits, name=qreg_name)
    creg = ClassicalRegister(circuit_qubits, name=creg_name)

    # Add the registers to circ_real
    circ_real.add_register(qreg)
    circ_real.add_register(creg)

    # Determine qubit and clbit indices for composition
    qubit_indices = list(range(circuit_qubits * repeat, circuit_qubits * (repeat + 1)))
    clbit_indices = list(range(circuit_qubits * repeat, circuit_qubits * (repeat + 1)))

    # Compose circ_combined into circ_real at the specified indices
    circ_real.compose(
        circ_combined,
        qubits=qubit_indices,
        clbits=clbit_indices,
        inplace=True
    )

print(f"Total qubits in circ_real after repetition: {circ_real.num_qubits}")
print(f"Total classical bits in circ_real after repetition: {circ_real.num_clbits}")

# Optional: Draw a portion of the real circuit to verify (commented out to avoid large plots)
circ_real.draw('mpl', scale=0.7)
plt.show()

# Transpile the real circuit with optimization level 2
pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)
transpiled_circ_real = pass_manager.run(circ_real)
print("Depth of transpiled circuit:", transpiled_circ_real.depth())

# Initialize the Sampler with options as a dictionary
sampler_options = {'shots': 1000}  # Include valid options
sampler = Sampler(backend=backend, options=sampler_options)

# Run the transpiled circuit on the real backend using the Sampler
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
if status == JobStatus.DONE:
    result = job.result()

    # Initialize real_counts dictionary
    real_counts = {view: defaultdict(float) for view in views}

    # Process the quasi-distribution
    quasi_dist = result.quasi_dists[0]

    # Convert quasi probabilities to probabilities
    probs = quasi_dist.nearest_probability_distribution()

    # Convert to counts
    counts = {int(k): v * sampler_options['shots'] for k, v in probs.items()}

    # Format the keys to bitstrings
    num_clbits = circ_real.num_clbits
    counts = {format(k, f'0{num_clbits}b'): v for k, v in counts.items()}

    # Process counts for each repetition and view
    for bitstring, value in counts.items():
        if not isinstance(value, (int, float)):
            print(f"Invalid count value for bitstring {bitstring}: {value}")
            continue
        for repeat in range(num_repeats):
            offset = repeat * circuit_qubits
            segment = bitstring[offset:offset + circuit_qubits]
            index = 0
            for view in views:
                view_bits = segment[index:index + 2]
                real_counts[view][view_bits] += value
                index += 2

    # Convert counts to regular dicts for plotting
    real_counts = {view: dict(count_dict) for view, count_dict in real_counts.items()}

    # Print real_counts to inspect the data structure
    print("real_counts:", real_counts)

    # Plot the results
    PlotCircuitResults(real_counts)
else:
    print("Job did not complete successfully.")
    # Retrieve and print job error message
    job_error_message = job.error_message()
    print(f"Job error message: {job_error_message}")

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