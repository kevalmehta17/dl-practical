# McCulloch-Pitts Neuron implementation

def mcp_neuron(inputs, weights, threshold):
    net = 0
    for i in range(len(inputs)):
        net += inputs[i] * weights[i]

    if net >= threshold:
        return 1
    else:
        return 0


# -------------------------
# AND Gate
# -------------------------
print("AND Gate")
weights_and = [1, 1]
threshold_and = 2

print("0 AND 0 =", mcp_neuron([0, 0], weights_and, threshold_and))
print("0 AND 1 =", mcp_neuron([0, 1], weights_and, threshold_and))
print("1 AND 0 =", mcp_neuron([1, 0], weights_and, threshold_and))
print("1 AND 1 =", mcp_neuron([1, 1], weights_and, threshold_and))


# -------------------------
# OR Gate
# -------------------------
print("\nOR Gate")
weights_or = [1, 1]
threshold_or = 1

print("0 OR 0 =", mcp_neuron([0, 0], weights_or, threshold_or))
print("0 OR 1 =", mcp_neuron([0, 1], weights_or, threshold_or))
print("1 OR 0 =", mcp_neuron([1, 0], weights_or, threshold_or))
print("1 OR 1 =", mcp_neuron([1, 1], weights_or, threshold_or))


# -------------------------
# NOT Gate
# -------------------------
print("\nNOT Gate")
weights_not = [-1]
threshold_not = 0

print("NOT 0 =", mcp_neuron([0], weights_not, threshold_not))
print("NOT 1 =", mcp_neuron([1], weights_not, threshold_not))
