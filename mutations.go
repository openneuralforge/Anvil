package blueprint

import (
	"encoding/json"
	"fmt"
	"math/rand"
)

// InsertNeuronOfTypeBetweenInputsAndOutputs inserts a new neuron of the specified type
// between all input and output nodes without removing existing connections.
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
	if bp.Debug {
		fmt.Printf("Inserted new Neuron with ID %d of type '%s' between inputs and outputs.\n", newNeuronID, neuronType)
	}
	// Connect input nodes to the new neuron
	for _, inputID := range bp.InputNodes {
		// Assign a random weight between -1 and 1
		weight := rand.Float64()*2 - 1
		newConnection := []float64{float64(inputID), weight}
		newNeuron.Connections = append(newNeuron.Connections, newConnection)
		if bp.Debug {
			fmt.Printf("Connected Input Neuron %d to New Neuron %d with weight %.4f.\n", inputID, newNeuronID, weight)
		}
	}

	// Connect the new neuron to each output neuron without removing existing connections
	for _, outputID := range bp.OutputNodes {
		outputNeuron, exists := bp.Neurons[outputID]
		if !exists {
			fmt.Printf("Warning: Output Neuron with ID %d does not exist.\n", outputID)
			continue
		}
		weight := rand.Float64()*2 - 1
		newConnection := []float64{float64(newNeuronID), weight}
		outputNeuron.Connections = append(outputNeuron.Connections, newConnection)
		if bp.Debug {
			fmt.Printf("Connected New Neuron %d to Output Neuron %d with weight %.4f.\n", newNeuronID, outputID, weight)
		}
	}

	// Initialize connection-dependent fields based on neuron type
	switch neuronType {
	case "lstm":
		bp.initializeLSTMWeights(newNeuron)
	case "nca":
		bp.initializeNCACustomFields(newNeuron)
	case "batch_norm":
		bp.initializeBatchNormFields(newNeuron)
		// Add cases for other neuron types as needed
	}

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
	if bp.Debug {
		fmt.Printf("Initialized GateWeights for LSTM Neuron %d with %d connections.\n", neuron.ID, numConnections)
	}
}

func (bp *Blueprint) createNeuron(id int, neuronType string) (*Neuron, error) {
	neuron := &Neuron{
		ID:          id,
		Type:        neuronType,
		Value:       rand.Float64()*2 - 1, // Random value between -1 and 1
		Bias:        rand.Float64()*2 - 1, // Random bias between -1 and 1
		Connections: [][]float64{},
		Activation:  "linear", // Default activation; will be overridden below
	}

	// Define possible activation functions
	activationFunctions := []string{"relu", "sigmoid", "tanh", "leaky_relu", "linear"}

	// Assign activation function based on type or randomly
	switch neuronType {
	case "dense":
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
	case "rnn":
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
	case "lstm":
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
		// Initialize gate weights for LSTM
		neuron.GateWeights = map[string][]float64{
			"input":  bp.RandomWeights(1), // Replace with actual connection size
			"forget": bp.RandomWeights(1),
			"output": bp.RandomWeights(1),
			"cell":   bp.RandomWeights(1),
		}
	case "cnn":
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
		// Initialize default kernels
		neuron.Kernels = [][]float64{
			{0.2, 0.5},
			{0.3, 0.4},
		}
	case "dropout":
		neuron.DropoutRate = 0.5 // Default dropout rate
	case "batch_norm":
		// Initialize BatchNormParams
		neuron.BatchNormParams = &BatchNormParams{
			Gamma: 1.0,
			Beta:  0.0,
			Mean:  0.0,
			Var:   1.0,
		}
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
	case "attention":
		neuron.Attention = true
		neuron.AttentionWeights = []float64{}
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
	case "nca":
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
		neuron.NCAState = make([]float64, 10)
		for i := range neuron.NCAState {
			neuron.NCAState[i] = rand.Float64()*2 - 1
		}
	default:
		neuron.Activation = activationFunctions[rand.Intn(len(activationFunctions))]
	}

	return neuron, nil
}

// isValidNeuronType checks if the provided neuron type is supported.
func (bp *Blueprint) isValidNeuronType(neuronType string) bool {
	supportedTypes := []string{
		"dense", "rnn", "lstm", "cnn", "dropout", "batch_norm", "attention", "nca",
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
		"dense", "rnn", "lstm", "cnn", "dropout", "batch_norm", "attention", "nca",
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

// initializeNCACustomFields initializes the NCA-specific fields for an NCA neuron.
func (bp *Blueprint) initializeNCACustomFields(neuron *Neuron) {
	// Example: Initialize NeighborhoodIDs and UpdateRules
	// For demonstration, let's assume NCA neurons are connected to all input neurons
	neuron.NeighborhoodIDs = append(neuron.NeighborhoodIDs, bp.InputNodes...)
	// Set a default update rule, e.g., "sum". This can be made configurable.
	neuron.UpdateRules = "sum"
	if bp.Debug {
		fmt.Printf("Initialized NCA-specific fields for NCA Neuron %d: NeighborhoodIDs=%v, UpdateRules=%s\n", neuron.ID, neuron.NeighborhoodIDs, neuron.UpdateRules)
	}
}

// blueprint.go

// initializeBatchNormFields initializes the BatchNormParams for a BatchNorm neuron.
// It sets Gamma, Beta, Mean, and Variance based on provided values or defaults.
func (bp *Blueprint) initializeBatchNormFields(neuron *Neuron) {
	// Check if BatchNormParams are already set
	if neuron.BatchNormParams != nil {
		fmt.Printf("BatchNorm Neuron %d: BatchNormParams already initialized.\n", neuron.ID)
		return
	}

	// Initialize BatchNormParams with default or specified values
	neuron.BatchNormParams = &BatchNormParams{
		Gamma: 1.0, // Scale parameter
		Beta:  0.0, // Shift parameter
		Mean:  0.0, // Running mean
		Var:   1.0, // Running variance
	}

	// Optionally, if you want to allow customization via JSON, you can check if values are provided
	// For simplicity, we're using default values here
	if bp.Debug {
		fmt.Printf("Initialized BatchNormParams for BatchNorm Neuron %d: Gamma=%.2f, Beta=%.2f, Mean=%.2f, Var=%.2f\n",
			neuron.ID, neuron.BatchNormParams.Gamma, neuron.BatchNormParams.Beta,
			neuron.BatchNormParams.Mean, neuron.BatchNormParams.Var)
	}
}

// InsertNeuronWithRandomConnectionsAndReconnect modifies the network by:
// - Appending a new neuron.
// - Randomly connecting it to existing neurons.
// - Reconnecting output neurons to the last `x` added neurons.
func (bp *Blueprint) InsertNeuronWithRandomConnectionsAndReconnect(neuronType string, reconnectToLastX int) error {
	// Validate the neuron type
	if !bp.isValidNeuronType(neuronType) {
		return fmt.Errorf("invalid neuron type: %s", neuronType)
	}

	// Generate a unique ID for the new neuron
	newNeuronID := bp.generateUniqueNeuronID()
	if newNeuronID == -1 {
		return fmt.Errorf("failed to generate a unique neuron ID")
	}

	// Create the new neuron
	newNeuron, err := bp.createNeuron(newNeuronID, neuronType)
	if err != nil {
		return fmt.Errorf("failed to create neuron of type '%s': %v", neuronType, err)
	}

	// Add the new neuron to the blueprint
	bp.Neurons[newNeuronID] = newNeuron
	if bp.Debug {
		fmt.Printf("Inserted new Neuron with ID %d of type '%s'.\n", newNeuronID, neuronType)
	}

	// Randomly connect the new neuron to other existing neurons
	neuronIDs := bp.getAllNeuronIDs()
	rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })
	numConnections := rand.Intn(2) + 1 // Randomly choose 1 or 2 connections

	for i := 0; i < numConnections && i < len(neuronIDs); i++ {
		targetID := neuronIDs[i]
		weight := rand.Float64()*2 - 1 // Random weight between -1 and 1
		newNeuron.Connections = append(newNeuron.Connections, []float64{float64(targetID), weight})
		if bp.Debug {
			fmt.Printf("Connected Neuron %d to existing Neuron %d with weight %.4f.\n", newNeuronID, targetID, weight)
		}
	}

	// Add the new neuron to the list of "active" neurons for future connections
	activeNeuronIDs := append(bp.getActiveNeuronIDs(), newNeuronID)

	// Reconnect all output neurons to the last `reconnectToLastX` neurons
	lastNeurons := getLastXNeurons(activeNeuronIDs, reconnectToLastX)
	for _, outputID := range bp.OutputNodes {
		outputNeuron, exists := bp.Neurons[outputID]
		if !exists {
			fmt.Printf("Warning: Output Neuron with ID %d does not exist.\n", outputID)
			continue
		}
		// Clear old connections for clean reconnection
		outputNeuron.Connections = nil
		for _, lastNeuronID := range lastNeurons {
			weight := rand.Float64()*2 - 1
			outputNeuron.Connections = append(outputNeuron.Connections, []float64{float64(lastNeuronID), weight})
			if bp.Debug {
				fmt.Printf("Reconnected Output Neuron %d to Neuron %d with weight %.4f.\n", outputID, lastNeuronID, weight)
			}
		}
	}

	return nil
}

// getAllNeuronIDs retrieves the IDs of all neurons in the blueprint.
func (bp *Blueprint) getAllNeuronIDs() []int {
	neuronIDs := []int{}
	for id := range bp.Neurons {
		neuronIDs = append(neuronIDs, id)
	}
	return neuronIDs
}

// getActiveNeuronIDs retrieves IDs of all neurons except inputs and outputs.
func (bp *Blueprint) getActiveNeuronIDs() []int {
	activeNeuronIDs := []int{}
	for id := range bp.Neurons {
		if !bp.isInputNode(id) && !bp.isOutputNode(id) {
			activeNeuronIDs = append(activeNeuronIDs, id)
		}
	}
	return activeNeuronIDs
}

// getLastXNeurons retrieves the last `x` neurons from the list, or fewer if not enough exist.
func getLastXNeurons(neuronIDs []int, x int) []int {
	if len(neuronIDs) <= x {
		return neuronIDs
	}
	return neuronIDs[len(neuronIDs)-x:]
}

/*
// InsertNeuronWithRandomConnections appends a new neuron to the network,
// randomly connects it to 1-2 existing neurons, and reconnects the outputs to maintain structure.
func (bp *Blueprint) InsertNeuronWithRandomConnections(neuronType string) error {
	// Validate the neuron type
	if !bp.isValidNeuronType(neuronType) {
		return fmt.Errorf("invalid neuron type: %s", neuronType)
	}

	// Generate a unique ID for the new neuron
	newNeuronID := bp.generateUniqueNeuronID()
	if newNeuronID == -1 {
		return fmt.Errorf("failed to generate a unique neuron ID")
	}

	// Create the new neuron
	newNeuron, err := bp.createNeuron(newNeuronID, neuronType)
	if err != nil {
		return fmt.Errorf("failed to create neuron of type '%s': %v", neuronType, err)
	}

	// Add the new neuron to the blueprint
	bp.Neurons[newNeuronID] = newNeuron
	if bp.Debug {
		fmt.Printf("Inserted new Neuron with ID %d of type '%s'.\n", newNeuronID, neuronType)
	}

	// Randomly connect the new neuron to 1-2 existing neurons
	neuronIDs := bp.getAllNeuronIDs()
	rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })
	numConnections := rand.Intn(2) + 1 // Randomly choose 1 or 2 connections

	for i := 0; i < numConnections && i < len(neuronIDs); i++ {
		targetID := neuronIDs[i]
		weight := rand.Float64()*2 - 1 // Random weight between -1 and 1
		newNeuron.Connections = append(newNeuron.Connections, []float64{float64(targetID), weight})
		if bp.Debug {
			fmt.Printf("Connected Neuron %d to existing Neuron %d with weight %.4f.\n", newNeuronID, targetID, weight)
		}
	}

	// Reconnect all output neurons to include the new neuron
	for _, outputID := range bp.OutputNodes {
		outputNeuron, exists := bp.Neurons[outputID]
		if !exists {
			fmt.Printf("Warning: Output Neuron with ID %d does not exist.\n", outputID)
			continue
		}
		weight := rand.Float64()*2 - 1
		newConnection := []float64{float64(newNeuronID), weight}
		outputNeuron.Connections = append(outputNeuron.Connections, newConnection)
		if bp.Debug {
			fmt.Printf("Connected New Neuron %d to Output Neuron %d with weight %.4f.\n", newNeuronID, outputID, weight)
		}
	}

	return nil
}*/

// InsertNeuronWithRandomConnections appends a new neuron to the network,
// randomly connects it to 1-2 existing neurons, and ensures selective output connections.
func (bp *Blueprint) InsertNeuronWithRandomConnections(neuronType string) error {
	// Validate the neuron type
	if !bp.isValidNeuronType(neuronType) {
		return fmt.Errorf("invalid neuron type: %s", neuronType)
	}

	// Generate a unique ID for the new neuron
	newNeuronID := bp.generateUniqueNeuronID()
	if newNeuronID == -1 {
		return fmt.Errorf("failed to generate a unique neuron ID")
	}

	// Create the new neuron
	newNeuron, err := bp.createNeuron(newNeuronID, neuronType)
	if err != nil {
		return fmt.Errorf("failed to create neuron of type '%s': %v", neuronType, err)
	}

	// Add the new neuron to the blueprint
	bp.Neurons[newNeuronID] = newNeuron
	if bp.Debug {
		fmt.Printf("Inserted new Neuron with ID %d of type '%s'.\n", newNeuronID, neuronType)
	}

	// Randomly connect the new neuron to 1-2 existing neurons
	neuronIDs := bp.getAllNeuronIDs()
	rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })
	numConnections := rand.Intn(2) + 1 // Randomly choose 1 or 2 connections

	for i := 0; i < numConnections && i < len(neuronIDs); i++ {
		targetID := neuronIDs[i]
		weight := rand.Float64()*2 - 1 // Random weight between -1 and 1
		newNeuron.Connections = append(newNeuron.Connections, []float64{float64(targetID), weight})
		if bp.Debug {
			fmt.Printf("Connected Neuron %d to existing Neuron %d with weight %.4f.\n", newNeuronID, targetID, weight)
		}
	}

	// Selectively connect the new neuron to output neurons
	if len(bp.OutputNodes) > 0 {
		selectedOutputID := bp.OutputNodes[rand.Intn(len(bp.OutputNodes))] // Randomly select one output neuron
		outputNeuron, exists := bp.Neurons[selectedOutputID]
		if exists {
			weight := rand.Float64()*2 - 1
			outputNeuron.Connections = append(outputNeuron.Connections, []float64{float64(newNeuronID), weight})
			if bp.Debug {
				fmt.Printf("Connected New Neuron %d to Output Neuron %d with weight %.4f.\n", newNeuronID, selectedOutputID, weight)
			}
		}
	}

	return nil
}
