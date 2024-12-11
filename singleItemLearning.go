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
						// Randomly decide the modification type
						modType := randomModificationType()

						var attemptResult NeuronAdditionAttempt

						switch modType {
						case "insert_neuron":
							// Randomly select a neuron type to attempt
							neuronType := neuronTypes[rand.Intn(len(neuronTypes))]

							// Serialize the current model
							modelJSON, err := bp.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing model: %v\n", workerID, err)
								continue
							}

							// Deserialize into a new Blueprint
							newBP := &Blueprint{}
							err = newBP.DeserializesFromJSON(modelJSON)
							if err != nil {
								fmt.Printf("Worker %d: Error deserializing model: %v\n", workerID, err)
								continue
							}

							// Attempt to insert a neuron of the selected type
							err = newBP.InsertNeuronWithRandomConnections(neuronType)
							if err != nil {
								fmt.Printf("Worker %d: Error inserting neuron of type '%s': %v\n", workerID, neuronType, err)
								continue
							}

							// Evaluate the new model on the single session
							tempSessions := []Session{sess}
							newExact, newGenerous, newForgive, _, _, _ :=
								newBP.EvaluateModelPerformance(tempSessions)

							// Calculate improvement on this session
							improvement := calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive)

							// Serialize the new model
							newModelJSON, err := newBP.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing new model: %v\n", workerID, err)
								continue
							}

							// If improvement is positive, send the attempt to the channel
							if improvement > 0 {
								attemptResult = NeuronAdditionAttempt{
									ModificationType: "insert_neuron",
									NeuronType:       neuronType,
									ModelJSON:        newModelJSON,
									ExactAcc:         newExact,
									GenerousAcc:      newGenerous,
									ForgiveAcc:       newForgive,
									Improvement:      improvement,
								}
								attemptCh <- attemptResult
							}

						case "add_connection":
							// Attempt to add a connection with random type
							sourceID, targetID := bp.getRandomConnectionPair()
							if sourceID == -1 || targetID == -1 {
								// No valid connection pair found
								continue
							}

							// Random weight between -1 and 1
							weight := rand.Float64()*2 - 1

							// Serialize the current model
							modelJSON, err := bp.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing model: %v\n", workerID, err)
								continue
							}

							// Deserialize into a new Blueprint
							newBP := &Blueprint{}
							err = newBP.DeserializesFromJSON(modelJSON)
							if err != nil {
								fmt.Printf("Worker %d: Error deserializing model: %v\n", workerID, err)
								continue
							}

							// Attempt to add the connection
							err = newBP.addConnection(sourceID, targetID, weight)
							if err != nil {
								fmt.Printf("Worker %d: Error adding connection (%d -> %d): %v\n", workerID, sourceID, targetID, err)
								continue
							}

							// Evaluate the new model on the single session
							tempSessions := []Session{sess}
							newExact, newGenerous, newForgive, _, _, _ :=
								newBP.EvaluateModelPerformance(tempSessions)

							// Calculate improvement on this session
							improvement := calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive)

							// Serialize the new model
							newModelJSON, err := newBP.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing new model: %v\n", workerID, err)
								continue
							}

							// If improvement is positive, send the attempt to the channel
							if improvement > 0 {
								attemptResult = NeuronAdditionAttempt{
									ModificationType: "add_connection",
									SourceID:         sourceID,
									TargetID:         targetID,
									Weight:           weight,
									ModelJSON:        newModelJSON,
									ExactAcc:         newExact,
									GenerousAcc:      newGenerous,
									ForgiveAcc:       newForgive,
									Improvement:      improvement,
								}
								attemptCh <- attemptResult
							}

						case "modify_activation":
							// Attempt to modify the activation function of a random neuron (non-input/output)
							neuronID := bp.getRandomHiddenNeuron()
							if neuronID == -1 {
								continue
							}

							// Randomly select a new activation function
							newActivation := randomActivationFunction()

							// Serialize the current model
							modelJSON, err := bp.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing model: %v\n", workerID, err)
								continue
							}

							// Deserialize into a new Blueprint
							newBP := &Blueprint{}
							err = newBP.DeserializesFromJSON(modelJSON)
							if err != nil {
								fmt.Printf("Worker %d: Error deserializing model: %v\n", workerID, err)
								continue
							}

							// Modify the activation function
							err = newBP.modifyActivationFunction(neuronID, newActivation)
							if err != nil {
								fmt.Printf("Worker %d: Error modifying activation function of neuron %d: %v\n", workerID, neuronID, err)
								continue
							}

							// Evaluate the new model on the single session
							tempSessions := []Session{sess}
							newExact, newGenerous, newForgive, _, _, _ :=
								newBP.EvaluateModelPerformance(tempSessions)

							// Calculate improvement on this session
							improvement := calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive)

							// Serialize the new model
							newModelJSON, err := newBP.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing new model: %v\n", workerID, err)
								continue
							}

							// If improvement is positive, send the attempt to the channel
							if improvement > 0 {
								attemptResult = NeuronAdditionAttempt{
									ModificationType: "modify_activation",
									NeuronType:       "", // Not applicable
									SourceID:         neuronID,
									TargetID:         0,   // Not applicable
									Weight:           0.0, // Not applicable
									Activation:       newActivation,
									ModelJSON:        newModelJSON,
									ExactAcc:         newExact,
									GenerousAcc:      newGenerous,
									ForgiveAcc:       newForgive,
									Improvement:      improvement,
								}
								attemptCh <- attemptResult
							}

						case "remove_connection":
							// Attempt to remove a random existing connection
							sourceID, targetID := bp.getRandomExistingConnectionPair()
							if sourceID == -1 || targetID == -1 {
								// No valid connection to remove
								continue
							}

							// Serialize the current model
							modelJSON, err := bp.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing model: %v\n", workerID, err)
								continue
							}

							// Deserialize into a new Blueprint
							newBP := &Blueprint{}
							err = newBP.DeserializesFromJSON(modelJSON)
							if err != nil {
								fmt.Printf("Worker %d: Error deserializing model: %v\n", workerID, err)
								continue
							}

							// Attempt to remove the connection
							newBP.removeConnection(sourceID, targetID)

							// Evaluate the new model on the single session
							tempSessions := []Session{sess}
							newExact, newGenerous, newForgive, _, _, _ :=
								newBP.EvaluateModelPerformance(tempSessions)

							// Calculate improvement on this session
							improvement := calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive)

							// Serialize the new model
							newModelJSON, err := newBP.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing new model: %v\n", workerID, err)
								continue
							}

							// If improvement is positive, send the attempt to the channel
							if improvement > 0 {
								attemptResult = NeuronAdditionAttempt{
									ModificationType: "remove_connection",
									SourceID:         sourceID,
									TargetID:         targetID,
									Weight:           0.0, // Not applicable
									ModelJSON:        newModelJSON,
									ExactAcc:         newExact,
									GenerousAcc:      newGenerous,
									ForgiveAcc:       newForgive,
									Improvement:      improvement,
								}
								attemptCh <- attemptResult
							}

						case "adjust_weight":
							// Attempt to adjust the weight of a random existing connection
							sourceID, targetID := bp.getRandomExistingConnectionPair()
							if sourceID == -1 || targetID == -1 {
								// No valid connection to adjust
								continue
							}

							// Randomly adjust the weight by a small delta
							delta := rand.Float64()*0.2 - 0.1 // Adjust by -0.1 to +0.1
							newWeight := bp.getConnectionWeight(sourceID, targetID) + delta

							// Serialize the current model
							modelJSON, err := bp.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing model: %v\n", workerID, err)
								continue
							}

							// Deserialize into a new Blueprint
							newBP := &Blueprint{}
							err = newBP.DeserializesFromJSON(modelJSON)
							if err != nil {
								fmt.Printf("Worker %d: Error deserializing model: %v\n", workerID, err)
								continue
							}

							// Attempt to adjust the weight
							err = newBP.addConnection(sourceID, targetID, newWeight) // Reuse addConnection to update weight
							if err != nil {
								fmt.Printf("Worker %d: Error adjusting weight for connection (%d -> %d): %v\n", workerID, sourceID, targetID, err)
								continue
							}

							// Evaluate the new model on the single session
							tempSessions := []Session{sess}
							newExact, newGenerous, newForgive, _, _, _ :=
								newBP.EvaluateModelPerformance(tempSessions)

							// Calculate improvement on this session
							improvement := calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive)

							// Serialize the new model
							newModelJSON, err := newBP.SerializeToJSON()
							if err != nil {
								fmt.Printf("Worker %d: Error serializing new model: %v\n", workerID, err)
								continue
							}

							// If improvement is positive, send the attempt to the channel
							if improvement > 0 {
								attemptResult = NeuronAdditionAttempt{
									ModificationType: "adjust_weight",
									SourceID:         sourceID,
									TargetID:         targetID,
									Weight:           newWeight,
									ModelJSON:        newModelJSON,
									ExactAcc:         newExact,
									GenerousAcc:      newGenerous,
									ForgiveAcc:       newForgive,
									Improvement:      improvement,
								}
								attemptCh <- attemptResult
							}

						default:
							// Unknown modification type
							continue
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

		// Collect all beneficial attempts for this batch
		var bestBatchAttempt *NeuronAdditionAttempt
		var bestBatchImprovement float64

		for attempt := range attemptCh {
			if attempt.Improvement > bestBatchImprovement {
				bestBatchImprovement = attempt.Improvement
				bestBatchAttempt = &attempt
			}
		}

		// After the batch, check if there was an improvement
		if bestBatchAttempt != nil && bestBatchImprovement > 0 {
			// Deserialize the best batch model
			err := bp.DeserializesFromJSON(bestBatchAttempt.ModelJSON)
			if err != nil {
				fmt.Printf("Batch %d: Error deserializing best batch model: %v\n", batchIdx, err)
				continue
			}

			// Re-evaluate the overall model
			newExact, newGenerous, newForgive, _, _, _ :=
				bp.EvaluateModelPerformance(sessions)

			// Calculate overall improvement
			overallImprovement := 0.0
			if newExact > initialExact {
				overallImprovement += newExact - initialExact
			}
			if newGenerous > initialGenerous {
				overallImprovement += newGenerous - initialGenerous
			}
			if newForgive > initialForgive {
				overallImprovement += newForgive - initialForgive
			}

			// Update the initial accuracies for the next batch
			initialExact, initialGenerous, initialForgive = newExact, newGenerous, newForgive

			// If overall improvement is positive, update the model
			if overallImprovement > 0 {
				fmt.Printf("\nBatch %d: Model improved by %.6f! Updating the main model.\n", batchIdx, overallImprovement)
				fmt.Printf("New Accuracies - Exact: %.6f%%, Generous: %.6f%%, Forgiveness: %.6f%%\n",
					newExact, newGenerous, newForgive)
			} else {
				fmt.Printf("\nBatch %d: No overall improvement after modifications.\n", batchIdx)
				// Optionally, revert changes or take other actions
			}
		} else {
			fmt.Printf("\nBatch %d: No beneficial modifications were found.\n", batchIdx)
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

// calculateImprovement calculates the total improvement based on accuracies.
func calculateImprovement(newExact, newGenerous, newForgive, initialExact, initialGenerous, initialForgive float64) float64 {
	improvement := 0.0
	if newExact > initialExact {
		improvement += newExact - initialExact
	}
	if newGenerous > initialGenerous {
		improvement += newGenerous - initialGenerous
	}
	if newForgive > initialForgive {
		improvement += newForgive - initialForgive
	}
	return improvement
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
