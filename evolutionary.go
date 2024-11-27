package blueprint

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
	"time"
)

// EvolutionaryTrain performs evolutionary training using neuroevolution.
func (bp *Blueprint) EvolutionaryTrain(sessions []Session, populationSize int, generations int, forgivenessThreshold float64) {
	rand.Seed(time.Now().UnixNano())

	// Initialize the population
	population := make([]*Blueprint, populationSize)
	for i := 0; i < populationSize; i++ {
		// Clone the blueprint and apply random mutations to weights and architecture
		individual := bp.Clone()
		individual.RandomizeWeights()
		individual.MutateArchitecture()
		population[i] = individual
	}

	for gen := 1; gen <= generations; gen++ {
		fmt.Printf("Generation %d\n", gen)

		// Evaluate each individual
		scores := make([]float64, populationSize)
		for i, individual := range population {
			exactAccuracy, generousAccuracy, forgivenessAccuracy, _, _, _ := individual.EvaluateModelPerformance(sessions, forgivenessThreshold)
			// Use a weighted sum of the accuracies as the fitness score
			scores[i] = (exactAccuracy + generousAccuracy + forgivenessAccuracy) / 3.0
		}

		// Select the best individuals
		bestIndividuals := selectBestIndividuals(population, scores, populationSize/2)

		// Generate new population through crossover and mutation
		newPopulation := make([]*Blueprint, populationSize)
		for i := 0; i < populationSize; i++ {
			parent1 := bestIndividuals[rand.Intn(len(bestIndividuals))]
			parent2 := bestIndividuals[rand.Intn(len(bestIndividuals))]
			child := parent1.Crossover(parent2)
			child.MutateWeights()
			child.MutateArchitecture()
			newPopulation[i] = child
		}

		population = newPopulation
	}

	// After the final generation, select the best individual
	bestIndividual := population[0]
	bestScore := 0.0
	for _, individual := range population {
		exactAccuracy, generousAccuracy, forgivenessAccuracy, _, _, _ := individual.EvaluateModelPerformance(sessions, forgivenessThreshold)
		score := (exactAccuracy + generousAccuracy + forgivenessAccuracy) / 3.0
		if score > bestScore {
			bestScore = score
			bestIndividual = individual
		}
	}

	// Update the original blueprint with the best found
	*bp = *bestIndividual

	fmt.Println("Evolutionary training completed. Best score:", bestScore)
}

// RandomizeWeights initializes weights and biases with random values
func (bp *Blueprint) RandomizeWeights() {
	for _, neuron := range bp.Neurons {
		// Skip input neurons
		if neuron.Type == "input" {
			continue
		}

		// Randomize biases
		neuron.Bias = rand.Float64()*2 - 1 // Random value between -1 and 1

		// Randomize connection weights
		for _, conn := range neuron.Connections {
			conn[1] = rand.Float64()*2 - 1 // Random value between -1 and 1
		}

		// Randomize gate weights for LSTM neurons
		if neuron.Type == "lstm" && neuron.GateWeights != nil {
			for gate, weights := range neuron.GateWeights {
				for i := range weights {
					weights[i] = rand.Float64()*2 - 1
				}
				neuron.GateWeights[gate] = weights
			}
		}
	}
}

// MutateWeights applies random perturbations to weights and biases
func (bp *Blueprint) MutateWeights() {
	mutationRate := 0.1 // Adjust as needed
	for _, neuron := range bp.Neurons {
		// Skip input neurons
		if neuron.Type == "input" {
			continue
		}

		// Mutate biases
		if rand.Float64() < mutationRate {
			neuron.Bias += rand.NormFloat64() * 0.1
		}

		// Mutate connection weights
		for _, conn := range neuron.Connections {
			if rand.Float64() < mutationRate {
				conn[1] += rand.NormFloat64() * 0.1
			}
		}

		// Mutate gate weights for LSTM neurons
		if neuron.Type == "lstm" && neuron.GateWeights != nil {
			for gate, weights := range neuron.GateWeights {
				for i := range weights {
					if rand.Float64() < mutationRate {
						weights[i] += rand.NormFloat64() * 0.1
					}
				}
				neuron.GateWeights[gate] = weights
			}
		}
	}
}

// MutateArchitecture randomly adds or removes neurons
func (bp *Blueprint) MutateArchitecture() {
	mutationRate := 0.05 // Adjust as needed

	// Possible neuron types to add
	neuronTypes := []string{"dense", "rnn", "lstm", "cnn", "dropout", "batch_norm", "attention", "nca"}

	if rand.Float64() < mutationRate {
		// Add a new neuron
		neuronType := neuronTypes[rand.Intn(len(neuronTypes))]
		err := bp.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
		if err != nil {
			fmt.Printf("Error adding neuron of type '%s': %v\n", neuronType, err)
		}
	}

	// Optionally remove a neuron
	if rand.Float64() < mutationRate && len(bp.Neurons) > len(bp.InputNodes)+len(bp.OutputNodes) {
		// Remove a random neuron that's not an input or output
		neuronIDs := []int{}
		for id := range bp.Neurons {
			if !bp.isInputNode(id) && !bp.isOutputNode(id) {
				neuronIDs = append(neuronIDs, id)
			}
		}
		if len(neuronIDs) > 0 {
			neuronIDToRemove := neuronIDs[rand.Intn(len(neuronIDs))]
			bp.RemoveNeuron(neuronIDToRemove)
			fmt.Printf("Removed Neuron with ID %d from the architecture.\n", neuronIDToRemove)
		}
	}
}

// RemoveNeuron removes a neuron and its associated connections
func (bp *Blueprint) RemoveNeuron(neuronID int) {
	delete(bp.Neurons, neuronID)

	// Remove connections to and from this neuron
	for _, neuron := range bp.Neurons {
		newConnections := [][]float64{}
		for _, conn := range neuron.Connections {
			sourceID := int(conn[0])
			if sourceID != neuronID {
				newConnections = append(newConnections, conn)
			}
		}
		neuron.Connections = newConnections
	}
}

// Crossover combines two parent blueprints to create a child blueprint
func (bp *Blueprint) Crossover(other *Blueprint) *Blueprint {
	child := bp.Clone()

	// For each neuron, randomly choose from parent1 or parent2
	for neuronID := range child.Neurons {
		if rand.Float64() < 0.5 {
			if neuron, exists := other.Neurons[neuronID]; exists {
				// Serialize the neuron to JSON
				data, err := json.Marshal(neuron)
				if err != nil {
					fmt.Printf("Error serializing neuron %d: %v\n", neuronID, err)
					continue
				}

				// Deserialize back into a new Neuron object
				var newNeuron Neuron
				err = json.Unmarshal(data, &newNeuron)
				if err != nil {
					fmt.Printf("Error deserializing neuron %d: %v\n", neuronID, err)
					continue
				}

				child.Neurons[neuronID] = &newNeuron
			}
		}
	}

	// Reinitialize activation functions for neurons if needed
	// Since neurons use activation function names, ensure the Blueprint's activation map is available

	return child
}

// Helper function to select the best individuals based on scores
func selectBestIndividuals(population []*Blueprint, scores []float64, num int) []*Blueprint {
	// Create a slice of indices
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}

	// Sort the indices based on scores in descending order
	sort.Slice(indices, func(i, j int) bool {
		return scores[indices[i]] > scores[indices[j]]
	})

	// Select the top individuals
	bestIndividuals := make([]*Blueprint, num)
	for i := 0; i < num; i++ {
		bestIndividuals[i] = population[indices[i]]
	}

	return bestIndividuals
}

func (bp *Blueprint) isOutputNode(neuronID int) bool {
	for _, id := range bp.OutputNodes {
		if id == neuronID {
			return true
		}
	}
	return false
}
