package blueprint

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Softmax activation function (applied across a slice)
func Softmax(inputs []float64) []float64 {
	max := inputs[0]
	for _, v := range inputs {
		if v > max {
			max = v
		}
	}
	expSum := 0.0
	expInputs := make([]float64, len(inputs))
	for i, v := range inputs {
		expInputs[i] = math.Exp(v - max)
		expSum += expInputs[i]
	}
	for i := range expInputs {
		expInputs[i] /= expSum
	}
	return expInputs
}

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
				if bp.Debug {
					fmt.Printf("CNN Neuron %d: No kernels provided. Initialized with default kernels.\n", cnnNeuron.ID)
				}
			}
			// Ensure activation is set to "relu"; if not, default to "relu"
			if cnnNeuron.Activation == "" {
				cnnNeuron.Activation = "relu"
				if bp.Debug {
					fmt.Printf("CNN Neuron %d: Activation not provided. Set to 'relu'.\n", cnnNeuron.ID)
				}
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
				if bp.Debug {
					fmt.Printf("BatchNorm Neuron %d: Activation not provided. Set to 'linear'.\n", bnNeuron.ID)
				}
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
				if bp.Debug {
					fmt.Printf("Neuron %d: Activation not provided. Set to 'linear'.\n", neuron.ID)
				}
			}
			bp.Neurons[neuron.ID] = &neuron
		}
	}

	return nil
}

// SaveToJSON saves the current Blueprint to a specified JSON file.
func (bp *Blueprint) SaveToJSON(fileName string) error {
	// Serialize the Blueprint to JSON
	data, err := json.MarshalIndent(bp, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to serialize Blueprint to JSON: %v", err)
	}

	// Write the JSON data to the specified file
	err = os.WriteFile(fileName, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write JSON to file '%s': %v", fileName, err)
	}

	fmt.Printf("Blueprint saved successfully to '%s'\n", fileName)
	return nil
}
