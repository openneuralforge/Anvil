package blueprint

import (
	"encoding/json"
	"fmt"
	"math/rand"
)

// InsertNeuronOfTypeBetweenInputsAndOutputs inserts a new neuron of the specified type
// between all input and output nodes. It updates the connections such that inputs connect
// to the new neuron, and the new neuron connects to the outputs. Existing direct connections
// from inputs to outputs are removed.
//
// Parameters:
// - neuronType: The type of neuron to insert (e.g., "dense", "rnn", "lstm", "cnn", etc.).
//
// Returns:
// - An error if the insertion fails.
func (bp *Blueprint) InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType string) error {
	// Validate the neuron type
	if !bp.isValidNeuronType(neuronType) {
		return fmt.Errorf("invalid neuron type: %s", neuronType)
	}

	// Generate a unique ID for the new neuron
	newNeuronID := bp.generateUniqueNeuronID()
	if newNeuronID == -1 {
		return fmt.Errorf("failed to generate a unique neuron ID")
	}

	// Create the new neuron based on the specified type
	newNeuron, err := bp.createNeuron(newNeuronID, neuronType)
	if err != nil {
		return fmt.Errorf("failed to create neuron of type '%s': %v", neuronType, err)
	}

	// Add the new neuron to the Blueprint
	bp.Neurons[newNeuronID] = newNeuron
	fmt.Printf("Inserted new Neuron with ID %d of type '%s' between inputs and outputs.\n", newNeuronID, neuronType)

	// Iterate over each output node to update connections
	for _, outputID := range bp.OutputNodes {
		outputNeuron, exists := bp.Neurons[outputID]
		if !exists {
			fmt.Printf("Warning: Output Neuron with ID %d does not exist.\n", outputID)
			continue
		}

		// Find and remove connections from input nodes to this output neuron
		var updatedConnections [][]float64
		var connectionsToRemove [][]float64
		for _, conn := range outputNeuron.Connections {
			sourceID := int(conn[0])
			if bp.isInputNode(sourceID) {
				connectionsToRemove = append(connectionsToRemove, conn)
			} else {
				updatedConnections = append(updatedConnections, conn)
			}
		}

		// Remove connections from input nodes to output neuron
		if len(connectionsToRemove) > 0 {
			outputNeuron.Connections = updatedConnections
			fmt.Printf("Removed %d connections from input nodes to Output Neuron %d.\n", len(connectionsToRemove), outputID)
		}

		// Connect input nodes to the new neuron
		for _, inputID := range bp.InputNodes {
			// Assign a random weight between -1 and 1
			weight := rand.Float64()*2 - 1
			newConnection := []float64{float64(inputID), weight}
			newNeuron.Connections = append(newNeuron.Connections, newConnection)
			fmt.Printf("Connected Input Neuron %d to New Neuron %d with weight %.4f.\n", inputID, newNeuronID, weight)
		}

		// Connect the new neuron to the output neuron
		weight := rand.Float64()*2 - 1
		newConnection := []float64{float64(newNeuronID), weight}
		outputNeuron.Connections = append(outputNeuron.Connections, newConnection)
		fmt.Printf("Connected New Neuron %d to Output Neuron %d with weight %.4f.\n", newNeuronID, outputID, weight)
	}

	// Initialize connection-dependent fields based on neuron type
	if neuronType == "lstm" {
		bp.initializeLSTMWeights(newNeuron)
	}
	// Add similar initializations for other neuron types if needed

	return nil
}

// initializeLSTMWeights initializes the GateWeights for an LSTM neuron based on its connections.
func (bp *Blueprint) initializeLSTMWeights(neuron *Neuron) {
	numConnections := len(neuron.Connections)
	if numConnections == 0 {
		fmt.Printf("Warning: LSTM Neuron %d has no connections to initialize GateWeights.\n", neuron.ID)
		return
	}

	// Initialize GateWeights with one weight per connection
	neuron.GateWeights = map[string][]float64{
		"input":  bp.RandomWeights(numConnections),
		"forget": bp.RandomWeights(numConnections),
		"output": bp.RandomWeights(numConnections),
		"cell":   bp.RandomWeights(numConnections),
	}

	fmt.Printf("Initialized GateWeights for LSTM Neuron %d with %d connections.\n", neuron.ID, numConnections)
}

// createNeuron initializes a neuron of the specified type with default or random values.
// It handles neuron-specific field initializations that are independent of connections.
func (bp *Blueprint) createNeuron(id int, neuronType string) (*Neuron, error) {
	neuron := &Neuron{
		ID:          id,
		Type:        neuronType,
		Value:       0.0,
		Bias:        0.0,
		Connections: [][]float64{},
		Activation:  "linear", // Default activation; can be overridden
	}

	switch neuronType {
	case "dense":
		neuron.Activation = "relu" // Default activation for dense neurons
	case "rnn":
		neuron.Activation = "tanh"
	case "lstm":
		neuron.Activation = "sigmoid"
		// GateWeights will be initialized based on connections
		// CellState is already initialized to 0.0 by default
	case "cnn":
		neuron.Activation = "relu"
		// Initialize default kernels if none are provided
		neuron.Kernels = [][]float64{
			{0.2, 0.5},
			{0.3, 0.4},
		}
	case "dropout":
		neuron.DropoutRate = 0.5 // Default dropout rate
	case "batch_norm":
		// Batch normalization parameters can be added here
	case "attention":
		neuron.Attention = true
	default:
		return nil, fmt.Errorf("unsupported neuron type: %s", neuronType)
	}

	return neuron, nil
}

// isValidNeuronType checks if the provided neuron type is supported.
func (bp *Blueprint) isValidNeuronType(neuronType string) bool {
	supportedTypes := []string{
		"dense", "rnn", "lstm", "cnn", "dropout", "batch_norm", "attention",
	}
	for _, t := range supportedTypes {
		if neuronType == t {
			return true
		}
	}
	return false
}

// generateUniqueNeuronID generates a unique neuron ID by finding the maximum existing ID and adding 1.
// Returns -1 if it fails to generate a unique ID.
func (bp *Blueprint) generateUniqueNeuronID() int {
	maxID := 0
	for id := range bp.Neurons {
		if id > maxID {
			maxID = id
		}
	}
	return maxID + 1
}

// isInputNode checks if a given neuron ID is an input node.
func (bp *Blueprint) isInputNode(neuronID int) bool {
	for _, id := range bp.InputNodes {
		if id == neuronID {
			return true
		}
	}
	return false
}

// MutateNetwork performs a series of mutations on the network.
// For demonstration, it inserts one neuron of each supported type between inputs and outputs.
func (bp *Blueprint) MutateNetwork() error {
	neuronTypes := []string{
		"dense", "rnn", "lstm", "cnn", "dropout", "batch_norm", "attention",
	}

	for _, neuronType := range neuronTypes {
		err := bp.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
		if err != nil {
			return fmt.Errorf("failed to insert neuron of type '%s': %v", neuronType, err)
		}
	}

	return nil
}

// ToJSON serializes the Blueprint to JSON for debugging or saving purposes.
func (bp *Blueprint) ToJSON() (string, error) {
	neurons := []Neuron{}
	for _, neuron := range bp.Neurons {
		neurons = append(neurons, *neuron)
	}

	data, err := json.MarshalIndent(neurons, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
