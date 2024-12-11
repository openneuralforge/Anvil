// singleItemLearning.go
package blueprint

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
)

// NeuronAdditionAttempt holds the result of a neuron or connection modification attempt.
type NeuronAdditionAttempt struct {
	ModificationType string  // "insert_neuron", "add_connection", "modify_activation", "remove_connection", "adjust_weight"
	NeuronType       string  // Applicable if ModificationType is "insert_neuron"
	SourceID         int     // Applicable if ModificationType is "add_connection", "remove_connection", or "adjust_weight"
	TargetID         int     // Applicable if ModificationType is "add_connection" or "remove_connection"
	Weight           float64 // Applicable if ModificationType is "add_connection" or "adjust_weight"
	Activation       string  // Applicable if ModificationType is "modify_activation"
	ModelJSON        string
	ExactAcc         float64
	GenerousAcc      float64
	ForgiveAcc       float64
	Improvement      float64
}

// LearnOneDataItemAtATime processes sessions in batches,
// attempts various modifications to improve performance on each session,
// and updates the main model if overall performance improves after each batch.
func (bp *Blueprint) LearnOneDataItemAtATime(
	sessions []Session,
	maxAttemptsPerSession int,
	neuronTypes []string,
	batchSize int, // Number of sessions to process at a time
) {
	fmt.Println("Starting LearnOneDataItemAtATime phase...")

	// Set default batch size if not specified or invalid
	if batchSize <= 0 {
		batchSize = 5
	}
	fmt.Printf("Batch size set to %d sessions.\n", batchSize)

	// Evaluate initial overall performance
	initialExact, initialGenerous, initialForgive, _, _, _ :=
		bp.EvaluateModelPerformance(sessions)

	// Determine the number of workers based on available CPU cores
	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	fmt.Printf("Utilizing %d worker(s) for modification attempts.\n", numWorkers)

	// Process sessions in batches
	for i := 0; i < len(sessions); i += batchSize {
		end := i + batchSize
		if end > len(sessions) {
			end = len(sessions)
		}
		batch := sessions[i:end]
		batchIdx := i/batchSize + 1

		fmt.Printf("\nProcessing Batch %d/%d...\n", batchIdx, (len(sessions)+batchSize-1)/batchSize)

		// Channel to collect beneficial attempts for this batch
		attemptCh := make(chan NeuronAdditionAttempt, len(batch)*maxAttemptsPerSession*5) // Adjust buffer as needed

		// WaitGroup for worker goroutines within the batch
		var wgWorkers sync.WaitGroup

		// Launch worker goroutines
		for w := 0; w < numWorkers; w++ {
			wgWorkers.Add(1)
			go func(workerID int) {
				defer wgWorkers.Done()
				for _, sess := range batch {
					for attempt := 0; attempt < maxAttemptsPerSession; attempt++ {
						// Perform random modification and evaluate the improvement
						attemptResult := bp.performRandomModification(sess, neuronTypes)

						// Send attempt to channel if it has improvement
						if attemptResult != nil {
							attemptCh <- *attemptResult
						}
					}
				}
			}(w + 1)
		}

		// Close the attempt channel once all workers are done
		go func() {
			wgWorkers.Wait()
			close(attemptCh)
		}()

		// Select the best attempt for the batch
		var bestBatchAttempt *NeuronAdditionAttempt
		var bestBatchImprovement float64

		for attempt := range attemptCh {
			if validateImprovement(
				attempt.ExactAcc, attempt.GenerousAcc, attempt.ForgiveAcc,
				initialExact, initialGenerous, initialForgive,
			) {
				improvement := calculateImprovement(
					attempt.ExactAcc, attempt.GenerousAcc, attempt.ForgiveAcc,
					initialExact, initialGenerous, initialForgive,
				)
				if improvement > bestBatchImprovement {
					bestBatchImprovement = improvement
					bestBatchAttempt = &attempt
				}
			}
		}

		// Update the model if the best attempt improves the performance
		if bestBatchAttempt != nil {
			// Create a new Blueprint from the best batch model
			newBlueprint := &Blueprint{}
			err := newBlueprint.DeserializesFromJSON(bestBatchAttempt.ModelJSON)
			if err != nil {
				fmt.Printf("Batch %d: Error deserializing best batch model: %v\n", batchIdx, err)
				continue
			}

			// Re-evaluate the overall model
			newExact, newGenerous, newForgive, _, _, _ :=
				newBlueprint.EvaluateModelPerformance(sessions)

			// Commit the update and adjust initial metrics
			if validateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive) {
				*bp = *newBlueprint // Update the main model with the new blueprint
				initialExact, initialGenerous, initialForgive = newExact, newGenerous, newForgive

				fmt.Printf("\nBatch %d: Model improved! Updating the main model.\n", batchIdx)
				fmt.Printf("New Accuracies - Exact: %.6f%%, Generous: %.6f%%, Forgiveness: %.6f%%\n",
					newExact, newGenerous, newForgive)
			} else {
				fmt.Printf("\nBatch %d: No beneficial modifications were found.\n", batchIdx)
			}
		}

	}

	fmt.Println("LearnOneDataItemAtATime phase completed.")
}

// randomModificationType randomly selects a modification type.
func randomModificationType() string {
	modTypes := []string{
		"insert_neuron",
		"add_connection",
		"modify_activation",
		"remove_connection",
		"adjust_weight",
	}
	return modTypes[rand.Intn(len(modTypes))]
}

// randomActivationFunction selects a random activation function.
func randomActivationFunction() string {
	activations := []string{
		"relu",
		"sigmoid",
		"tanh",
		"leaky_relu",
		"softmax",
	}
	return activations[rand.Intn(len(activations))]
}

// calculateImprovement ensures at least one metric improves without others degrading.
func calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive float64) float64 {
	return (newExact - initialExact) + (newGenerous - initialGenerous) + (newForgive - initialForgive)
}

// validateImprovement checks if at least one metric improved and no metrics degraded.
func validateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive float64) bool {
	return (newExact >= initialExact && newGenerous >= initialGenerous && newForgive >= initialForgive) &&
		(newExact > initialExact || newGenerous > initialGenerous || newForgive > initialForgive)
}

// getRandomHiddenNeuron selects a random hidden neuron (non-input, non-output).
// Returns -1 if no hidden neuron is found.
func (bp *Blueprint) getRandomHiddenNeuron() int {
	hiddenNeurons := []int{}
	for id, neuron := range bp.Neurons {
		if neuron.Type != "input" && neuron.Type != "output" {
			hiddenNeurons = append(hiddenNeurons, id)
		}
	}
	if len(hiddenNeurons) == 0 {
		return -1
	}
	return hiddenNeurons[rand.Intn(len(hiddenNeurons))]
}

// getRandomExistingConnectionPair selects a random existing connection pair.
// Returns -1, -1 if no existing connection is found.
func (bp *Blueprint) getRandomExistingConnectionPair() (int, int) {
	existingConnections := [][]float64{}
	for sourceID, neuron := range bp.Neurons {
		for _, conn := range neuron.Connections {
			targetID := int(conn[0])
			existingConnections = append(existingConnections, []float64{float64(sourceID), float64(targetID)})
		}
	}
	if len(existingConnections) == 0 {
		return -1, -1
	}
	selected := existingConnections[rand.Intn(len(existingConnections))]
	return int(selected[0]), int(selected[1])
}

// modifyActivationFunction changes the activation function of a neuron.
func (bp *Blueprint) modifyActivationFunction(neuronID int, newActivation string) error {
	neuron, exists := bp.Neurons[neuronID]
	if !exists {
		return fmt.Errorf("neuron ID %d does not exist", neuronID)
	}
	neuron.Activation = newActivation
	return nil
}

// getConnectionWeight retrieves the weight of a connection between sourceID and targetID.
// Returns 0.0 if connection does not exist.
func (bp *Blueprint) getConnectionWeight(sourceID, targetID int) float64 {
	sourceNeuron, exists := bp.Neurons[sourceID]
	if !exists {
		return 0.0
	}
	for _, conn := range sourceNeuron.Connections {
		if int(conn[0]) == targetID {
			return conn[1]
		}
	}
	return 0.0
}

// performRandomModification executes a random modification and evaluates its impact.
func (bp *Blueprint) performRandomModification(sess Session, neuronTypes []string) *NeuronAdditionAttempt {
	// Randomly decide the modification type
	modType := randomModificationType()

	// Serialize the current model
	modelJSON, err := bp.SerializeToJSON()
	if err != nil {
		fmt.Printf("Error serializing model: %v\n", err)
		return nil
	}

	// Deserialize into a new Blueprint
	newBP := &Blueprint{}
	err = newBP.DeserializesFromJSON(modelJSON)
	if err != nil {
		fmt.Printf("Error deserializing model: %v\n", err)
		return nil
	}

	// Perform the modification
	switch modType {
	case "insert_neuron":
		neuronType := neuronTypes[rand.Intn(len(neuronTypes))]
		err = newBP.InsertNeuronWithRandomConnections(neuronType)
	case "add_connection":
		sourceID, targetID := bp.getRandomConnectionPair()
		if sourceID != -1 && targetID != -1 {
			err = newBP.addConnection(sourceID, targetID, rand.Float64()*2-1)
		}
	case "modify_activation":
		neuronID := bp.getRandomHiddenNeuron()
		if neuronID != -1 {
			err = newBP.modifyActivationFunction(neuronID, randomActivationFunction())
		}
	case "remove_connection":
		sourceID, targetID := bp.getRandomExistingConnectionPair()
		if sourceID != -1 && targetID != -1 {
			newBP.removeConnection(sourceID, targetID)
		}
	case "adjust_weight":
		sourceID, targetID := bp.getRandomExistingConnectionPair()
		if sourceID != -1 && targetID != -1 {
			err = newBP.addConnection(sourceID, targetID, bp.getConnectionWeight(sourceID, targetID)+(rand.Float64()*0.2-0.1))
		}
	}

	if err != nil {
		return nil
	}

	// Evaluate the new model
	tempSessions := []Session{sess}
	newExact, newGenerous, newForgive, _, _, _ :=
		newBP.EvaluateModelPerformance(tempSessions)

	improvement := calculateImprovement(newExact, newGenerous, newForgive, 0, 0, 0) // Improvement per session

	if improvement > 0 {
		return &NeuronAdditionAttempt{
			ModificationType: modType,
			ModelJSON:        modelJSON,
			ExactAcc:         newExact,
			GenerousAcc:      newGenerous,
			ForgiveAcc:       newForgive,
			Improvement:      improvement,
		}
	}

	return nil
}
