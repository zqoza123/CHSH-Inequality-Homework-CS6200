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

circuits = [MeasureWithView(BellCircuit('q' + view, 'c' + view), view) for view in views]

for circuit, view in zip(circuits, views):
    print(f"Circuit for view {view}:")
    circuit.draw(output='mpl')
    plt.show()

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

def PlotCircuitResults(circuit_results: dict):
    try:
        formatted_results = {}
        for view, counts in circuit_results.items():
            if isinstance(counts, dict):
                formatted_counts = {}
                for k, v in counts.items():
                    if isinstance(v, dict):
                        print(f"Skipping nested dict for key {k} in view {view}.")
                        continue
                    if not isinstance(k, str):
                        k = str(k)
                    if not isinstance(v, (int, float)):
                        print(f"Invalid count value for key {k} in view {view}: {v}")
                        continue
                    formatted_counts[k] = v
                if formatted_counts:
                    formatted_results[view] = formatted_counts
            else:
                print(f"Counts for view {view} are not in the expected format.")
        plot_histogram(formatted_results, figsize=(8, 6))
        plt.show()
    except Exception as e:
        print(f"Error encountered while plotting results: {e}")

PlotCircuitResults({
    "AB": {"00": 400, "01": 300, "10": 200, "11": 100},
    "Ab": {"00": 100, "01": 200, "10": 300, "11": 400},
    "aB": {"00": 250, "01": 250, "10": 250, "11": 250},
    "ab": {"00": 900, "01": 0, "10": 0, "11": 100},
})

simulator = AerSimulator()
job = simulator.run(circuits, shots=1000)
ideal_results = job.result()

ideal_counts = {view: ideal_results.get_counts(circuit) for view, circuit in zip(views, circuits)}

PlotCircuitResults(ideal_counts)

service = QiskitRuntimeService()

backend_name = 'ibm_kyiv'
backend = service.backend(backend_name)

noise_model = NoiseModel.from_backend(backend)

noisy_simulator = AerSimulator(noise_model=noise_model)

transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)
transpiled_circuits = transpiler.run(circuits)

job = noisy_simulator.run(transpiled_circuits, shots=1000)
noisy_results = job.result()

noisy_counts = {view: noisy_results.get_counts(circuit) for view, circuit in zip(views, transpiled_circuits)}

PlotCircuitResults(noisy_counts)



circ_combined = QuantumCircuit()

for view in views:
    qreg_name = f'q_{view}'
    creg_name = f'c_{view}'

    experiment_circuit = MeasureWithView(BellCircuit(qreg_name, creg_name), view)

    circ_combined.add_register(experiment_circuit.qregs[0])
    circ_combined.add_register(experiment_circuit.cregs[0])

    qubit_indices = list(range(circ_combined.num_qubits - 2, circ_combined.num_qubits))
    clbit_indices = list(range(circ_combined.num_clbits - 2, circ_combined.num_clbits))

    circ_combined.compose(
        experiment_circuit,
        qubits=qubit_indices,
        clbits=clbit_indices,
        inplace=True
    )

circ_combined.draw('mpl')
plt.show()

num_qubits = 8 * ((backend.num_qubits * 4 // 5) // 8)
print(f"Total qubits in circ_real: {num_qubits}")

circuit_qubits = circ_combined.num_qubits
num_repeats = num_qubits // circuit_qubits
print(f"Number of repeats: {num_repeats}")

circ_real = QuantumCircuit()

for repeat in range(num_repeats):
    qreg_name = f'q_rep{repeat}'
    creg_name = f'c_rep{repeat}'

    qreg = QuantumRegister(circuit_qubits, name=qreg_name)
    creg = ClassicalRegister(circuit_qubits, name=creg_name)

    circ_real.add_register(qreg)
    circ_real.add_register(creg)

    qubit_indices = list(range(circuit_qubits * repeat, circuit_qubits * (repeat + 1)))
    clbit_indices = list(range(circuit_qubits * repeat, circuit_qubits * (repeat + 1)))

    circ_real.compose(
        circ_combined,
        qubits=qubit_indices,
        clbits=clbit_indices,
        inplace=True
    )

print(f"Total qubits in circ_real after repetition: {circ_real.num_qubits}")
print(f"Total classical bits in circ_real after repetition: {circ_real.num_clbits}")

circ_real.draw('mpl', scale=0.7)
plt.show()

pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)
transpiled_circ_real = pass_manager.run(circ_real)
print("Depth of transpiled circuit:", transpiled_circ_real.depth())

sampler_options = {'shots': 1000}  # Include valid options
sampler = Sampler(backend=backend, options=sampler_options)

job = sampler.run([transpiled_circ_real])
job_id = job.job_id()
print(f"Job ID: {job_id}")

import time

while True:
    status = job.status()
    print(f"Job status: {status}")
    if status == JobStatus.DONE or status == JobStatus.ERROR:
        break
    time.sleep(10)

if status == JobStatus.DONE:
    result = job.result()

    real_counts = {view: defaultdict(float) for view in views}

    quasi_dist = result.quasi_dists[0]

    probs = quasi_dist.nearest_probability_distribution()

    counts = {int(k): v * sampler_options['shots'] for k, v in probs.items()}

    num_clbits = circ_real.num_clbits
    counts = {format(k, f'0{num_clbits}b'): v for k, v in counts.items()}

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

    real_counts = {view: dict(count_dict) for view, count_dict in real_counts.items()}

    print("real_counts:", real_counts)

    PlotCircuitResults(real_counts)
else:
    print("Job did not complete successfully.")
    job_error_message = job.error_message()
    print(f"Job error message: {job_error_message}")

circuit_qubits = circ_combined.num_qubits
num_repeats = num_qubits // circuit_qubits

if num_repeats == 0:
    num_repeats = 1

print(f"Number of repeats: {num_repeats}")

circ_real = QuantumCircuit()
for repeat in range(num_repeats):
    offset = repeat * circuit_qubits
    circ_real.add_register(QuantumRegister(circuit_qubits, name=f'q{repeat}'))
    circ_real.add_register(ClassicalRegister(circuit_qubits, name=f'c{repeat}'))
    circ_real.compose(
        circ_combined,
        qubits=range(offset, offset + circuit_qubits),
        clbits=range(offset, offset + circuit_qubits),
        inplace=True
    )

circ_real.draw('mpl')
plt.show()

pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)  # Corrected function call
transpiled_circ_real = pass_manager.run(circ_real)
print("Depth of transpiled circuit:", transpiled_circ_real.depth())

sampler_options = SamplerOptions(resilience_level=1)
sampler = Sampler(backend=backend, options=sampler_options)

job = sampler.run([transpiled_circ_real])
job_id = job.job_id()
print(f"Job ID: {job_id}")

import time
while True:
    status = job.status()
    print(f"Job status: {status}")
    if status == JobStatus.DONE or status == JobStatus.ERROR:
        break
    time.sleep(10)

result = job.result()

real_counts = {view: defaultdict(float) for view in views}

quasi_dist = result.quasi_dists[0]

total_shots = 1000

probs = quasi_dist.nearest_probability_distribution()

counts = {int(k): v * total_shots for k, v in probs.items()}

num_clbits = circ_real.num_clbits
counts = {format(k, f'0{num_clbits}b'): v for k, v in counts.items()}

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

real_counts = {view: dict(count_dict) for view, count_dict in real_counts.items()}

print("real_counts:", real_counts)

PlotCircuitResults(real_counts)


def ExpectationValue(counts):

    total_shots = sum(counts.values())

    if total_shots == 0:
        raise ValueError("Total number of shots is zero. Cannot compute expectation value.")

    P00 = counts.get('00', 0) / total_shots
    P11 = counts.get('11', 0) / total_shots
    P01 = counts.get('01', 0) / total_shots
    P10 = counts.get('10', 0) / total_shots

    expectation = P00 + P11 - P01 - P10

    return expectation


ideal_expectation = {}

for view in views:
    counts = ideal_counts.get(view, {})
    expectation = ExpectationValue(counts)
    ideal_expectation[view] = expectation
    print(f"Expectation value for {view} (Ideal Simulator): {expectation:.4f}")

print("\n")


noisy_expectation = {}

for view in views:
    counts = noisy_counts.get(view, {})
    expectation = ExpectationValue(counts)
    noisy_expectation[view] = expectation
    print(f"Expectation value for {view} (Noisy Simulator): {expectation:.4f}")

print("\n")

real_expectation = {}

for view in views:
    counts = real_counts.get(view, {})
    expectation = ExpectationValue(counts)
    real_expectation[view] = expectation
    print(f"Expectation value for {view} (IBM Quantum Computer): {expectation:.4f}")

def CHSHValue(ev):

    try:
        AB = ev.get('AB', 0)
        Ab = ev.get('Ab', 0)
        aB = ev.get('aB', 0)
        ab = ev.get('ab', 0)
        CHSH = AB + Ab + aB - ab

        return CHSH
    except Exception as e:
        print(f"Error calculating CHSH value: {e}")
        return None



ideal_CHSH = CHSHValue(ideal_expectation)

print(f"CHSH Value from Ideal Simulator: {ideal_CHSH:.4f}")
print(f"CHSH Inequality Violated: {ideal_CHSH > 2}")

print("\n")

noisy_CHSH = CHSHValue(noisy_expectation)

print(f"CHSH Value from Noisy Simulator: {noisy_CHSH:.4f}")
print(f"CHSH Inequality Violated: {noisy_CHSH > 2}")

print("\n")


real_CHSH = CHSHValue(real_expectation)

print(f"CHSH Value from IBM Quantum Computer: {real_CHSH:.4f}")
print(f"CHSH Inequality Violated: {real_CHSH > 2}")