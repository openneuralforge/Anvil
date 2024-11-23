package blueprint

import (
	"encoding/json"
	"fmt"
	"math/rand"
)

// Blueprint encapsulates the entire neural network
type Blueprint struct {
	Neurons             map[int]*Neuron
	QuantumNeurons      map[int]*QuantumNeuron
	InputNodes          []int
	OutputNodes         []int
	ScalarActivationMap map[string]ActivationFunc
}

// NewBlueprint initializes a new Blueprint
func NewBlueprint() *Blueprint {
	return &Blueprint{
		Neurons:             make(map[int]*Neuron),
		QuantumNeurons:      make(map[int]*QuantumNeuron),
		ScalarActivationMap: scalarActivationFunctions,
	}
}

// blueprint.go

// LoadNeurons loads neurons from a JSON string
func (bp *Blueprint) LoadNeurons(jsonData string) error {
	var rawNeurons []json.RawMessage
	if err := json.Unmarshal([]byte(jsonData), &rawNeurons); err != nil {
		return err
	}

	for _, rawNeuron := range rawNeurons {
		var baseNeuron struct {
			ID   int    `json:"id"`
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rawNeuron, &baseNeuron); err != nil {
			return err
		}

		switch baseNeuron.Type {
		case "quantum":
			var qNeuron QuantumNeuron
			if err := json.Unmarshal(rawNeuron, &qNeuron); err != nil {
				return err
			}
			bp.QuantumNeurons[qNeuron.ID] = &qNeuron

		case "nca":
			var ncaNeuron Neuron
			if err := json.Unmarshal(rawNeuron, &ncaNeuron); err != nil {
				return err
			}
			bp.Neurons[ncaNeuron.ID] = &ncaNeuron

		case "cnn":
			var cnnNeuron Neuron
			if err := json.Unmarshal(rawNeuron, &cnnNeuron); err != nil {
				return err
			}
			// Ensure kernels are initialized; if not provided, initialize with default kernels
			if len(cnnNeuron.Kernels) == 0 {
				// Example: Initialize with default kernels
				cnnNeuron.Kernels = [][]float64{
					{0.2, 0.5}, // Default Kernel 0
					{0.3, 0.4}, // Default Kernel 1
				}
				fmt.Printf("CNN Neuron %d: No kernels provided. Initialized with default kernels.\n", cnnNeuron.ID)
			}
			// Ensure activation is set to "relu"; if not, default to "relu"
			if cnnNeuron.Activation == "" {
				cnnNeuron.Activation = "relu"
				fmt.Printf("CNN Neuron %d: Activation not provided. Set to 'relu'.\n", cnnNeuron.ID)
			}
			bp.Neurons[cnnNeuron.ID] = &cnnNeuron

		case "batch_norm":
			var bnNeuron Neuron
			if err := json.Unmarshal(rawNeuron, &bnNeuron); err != nil {
				return err
			}
			// Initialize BatchNormParams
			bnNeuron.BatchNormParams = &BatchNormParams{
				Gamma: 1.0,
				Beta:  0.0,
				Mean:  0.0,
				Var:   1.0,
			}
			// Ensure activation is set; default to "linear" if not provided
			if bnNeuron.Activation == "" {
				bnNeuron.Activation = "linear"
				fmt.Printf("BatchNorm Neuron %d: Activation not provided. Set to 'linear'.\n", bnNeuron.ID)
			}
			bp.Neurons[bnNeuron.ID] = &bnNeuron

		default:
			var neuron Neuron
			if err := json.Unmarshal(rawNeuron, &neuron); err != nil {
				return err
			}
			// Initialize gate weights for LSTM neurons
			if neuron.Type == "lstm" {
				neuron.GateWeights = map[string][]float64{
					"input":  bp.RandomWeights(len(neuron.Connections)),
					"forget": bp.RandomWeights(len(neuron.Connections)),
					"output": bp.RandomWeights(len(neuron.Connections)),
					"cell":   bp.RandomWeights(len(neuron.Connections)),
				}
			}
			// Ensure activation is set; default to "linear" if not provided
			if neuron.Activation == "" {
				neuron.Activation = "linear"
				fmt.Printf("Neuron %d: Activation not provided. Set to 'linear'.\n", neuron.ID)
			}
			bp.Neurons[neuron.ID] = &neuron
		}
	}

	return nil
}

// RandomWeights generates random weights for connections
func (bp *Blueprint) RandomWeights(size int) []float64 {
	weights := make([]float64, size)
	for i := range weights {
		weights[i] = rand.NormFloat64() * 0.5 // Increase scale
	}
	return weights
}

// AddInputNodes adds multiple input nodes to the network
func (bp *Blueprint) AddInputNodes(ids []int) {
	bp.InputNodes = append(bp.InputNodes, ids...)
}

// AddOutputNodes adds multiple output nodes to the network
func (bp *Blueprint) AddOutputNodes(ids []int) {
	bp.OutputNodes = append(bp.OutputNodes, ids...)
}

// ApplyScalarActivation applies the specified scalar activation function
func (bp *Blueprint) ApplyScalarActivation(value float64, activation string) float64 {
	if actFunc, exists := bp.ScalarActivationMap[activation]; exists {
		return actFunc(value)
	}
	// Log a warning and use linear activation
	fmt.Printf("Warning: Undefined activation '%s'. Using linear activation.\n", activation)
	return Linear(value)
}

// Forward propagates inputs through the network
func (bp *Blueprint) Forward(inputs map[int]float64, timesteps int) {
	// Set input neurons
	for id, value := range inputs {
		if neuron, exists := bp.Neurons[id]; exists {
			neuron.Value = value
			fmt.Printf("Input Neuron %d set to %f\n", id, value)
		}
	}

	// Process neurons over timesteps for recurrent networks
	for t := 0; t < timesteps; t++ {
		fmt.Printf("=== Timestep %d ===\n", t)
		// Process neurons in order of IDs for simplicity
		for id := 1; id <= len(bp.Neurons); id++ {
			neuron, exists := bp.Neurons[id]
			if !exists {
				continue
			}

			// Skip input neurons
			if neuron.Type == "input" {
				continue
			}

			// Gather inputs from connected neurons
			inputValues := []float64{}
			for _, conn := range neuron.Connections {
				sourceID := int(conn[0])
				weight := conn[1]
				if sourceNeuron, exists := bp.Neurons[sourceID]; exists {
					inputValues = append(inputValues, sourceNeuron.Value*weight)
				}
			}

			// Special handling for attention mechanisms
			if neuron.Type == "attention" {
				attentionWeights := bp.ComputeAttentionWeights(neuron, inputValues)
				bp.ApplyAttention(neuron, inputValues, attentionWeights)
			} else {
				bp.ProcessNeuron(neuron, inputValues, t)
			}
		}
	}

	// Apply Softmax to Output Layer
	bp.ApplySoftmax()
}

// RunNetwork runs the neural network with given inputs and timesteps
func (bp *Blueprint) RunNetwork(inputs map[int]float64, timesteps int) {
	bp.Forward(inputs, timesteps)
	outputs := bp.GetOutputs()
	fmt.Println("Final Outputs:")
	for id, value := range outputs {
		fmt.Printf("Neuron %d: %f\n", id, value)
	}
}

// GetOutputs retrieves the output values from the network
func (bp *Blueprint) GetOutputs() map[int]float64 {
	outputs := make(map[int]float64)
	for _, id := range bp.OutputNodes {
		if neuron, exists := bp.Neurons[id]; exists {
			outputs[id] = neuron.Value
		}
	}
	return outputs
}
