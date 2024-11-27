// nas.go
package blueprint

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
)

// SimpleNAS performs a basic neural architecture search by incrementally adding one neuron at a time
// and keeping the change if it improves the model's evaluation on any of the three evaluation metrics.
func (bp *Blueprint) SimpleNAS(sessions []Session, maxIterations int, forgivenessThreshold float64) {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Keep track of the best model and its performance
	bestBlueprint := bp.Clone() // Assume we have a Clone method
	bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy, _, _, _ := bestBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

	fmt.Printf("Initial model performance: Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
		bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy)

	for iteration := 1; iteration <= maxIterations; iteration++ {
		// Clone the best blueprint to create a new candidate
		candidateBlueprint := bestBlueprint.Clone()

		// Randomly select a neuron type to add
		neuronTypes := []string{"dense", "rnn", "lstm", "cnn", "dropout", "batch_norm", "attention", "nca"}
		neuronType := neuronTypes[rand.Intn(len(neuronTypes))]

		// Insert a neuron of this type between inputs and outputs
		err := candidateBlueprint.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
		if err != nil {
			fmt.Printf("Iteration %d: Failed to insert neuron of type '%s': %v\n", iteration, neuronType, err)
			continue
		}

		// Evaluate the candidate model
		exactAccuracy, generousAccuracy, forgivenessAccuracy, _, _, _ := candidateBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

		// Check if the candidate model improves on any of the three metrics
		if exactAccuracy > bestExactAccuracy || generousAccuracy > bestGenerousAccuracy || forgivenessAccuracy > bestForgivenessAccuracy {
			// Update the best model
			bestBlueprint = candidateBlueprint
			bestExactAccuracy = exactAccuracy
			bestGenerousAccuracy = generousAccuracy
			bestForgivenessAccuracy = forgivenessAccuracy

			fmt.Printf("Iteration %d: Improved model found! Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
				iteration, exactAccuracy, generousAccuracy, forgivenessAccuracy)
		} else {
			fmt.Printf("Iteration %d: No improvement.\n", iteration)
		}
	}

	// Update the original blueprint with the best found
	*bp = *bestBlueprint
}

// Clone creates a deep copy of the Blueprint using JSON serialization
func (bp *Blueprint) Clone() *Blueprint {
	// Serialize the blueprint to JSON
	data, err := json.Marshal(bp)
	if err != nil {
		fmt.Printf("Error serializing blueprint: %v\n", err)
		return nil
	}

	// Deserialize the JSON back into a new Blueprint object
	var newBP Blueprint
	err = json.Unmarshal(data, &newBP)
	if err != nil {
		fmt.Printf("Error deserializing blueprint: %v\n", err)
		return nil
	}

	// Reinitialize any nil maps or function maps
	if newBP.Neurons == nil {
		newBP.Neurons = make(map[int]*Neuron)
	}
	if newBP.ScalarActivationMap == nil {
		newBP.InitializeActivationFunctions()
	}

	return &newBP
}

// DeepCopy creates a deep copy of a Neuron
func (neuron *Neuron) DeepCopy() *Neuron {
	newNeuron := *neuron // Shallow copy

	// Deep copy slices and maps within neuron
	if neuron.Connections != nil {
		newNeuron.Connections = make([][]float64, len(neuron.Connections))
		for i, conn := range neuron.Connections {
			newConn := make([]float64, len(conn))
			copy(newConn, conn)
			newNeuron.Connections[i] = newConn
		}
	}

	if neuron.GateWeights != nil {
		newNeuron.GateWeights = make(map[string][]float64)
		for gate, weights := range neuron.GateWeights {
			newWeights := make([]float64, len(weights))
			copy(newWeights, weights)
			newNeuron.GateWeights[gate] = newWeights
		}
	}

	if neuron.BatchNormParams != nil {
		newParams := *neuron.BatchNormParams
		newNeuron.BatchNormParams = &newParams
	}

	if neuron.Kernels != nil {
		newNeuron.Kernels = make([][]float64, len(neuron.Kernels))
		for i, kernel := range neuron.Kernels {
			newKernel := make([]float64, len(kernel))
			copy(newKernel, kernel)
			newNeuron.Kernels[i] = newKernel
		}
	}

	if neuron.NCAState != nil {
		newNeuron.NCAState = make([]float64, len(neuron.NCAState))
		copy(newNeuron.NCAState, neuron.NCAState)
	}

	if neuron.AttentionWeights != nil {
		newNeuron.AttentionWeights = make([]float64, len(neuron.AttentionWeights))
		copy(newNeuron.AttentionWeights, neuron.AttentionWeights)
	}

	if neuron.NeighborhoodIDs != nil {
		newNeuron.NeighborhoodIDs = make([]int, len(neuron.NeighborhoodIDs))
		copy(newNeuron.NeighborhoodIDs, neuron.NeighborhoodIDs)
	}

	return &newNeuron
}
