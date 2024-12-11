package blueprint

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
)

// ConnectionAttempt holds the result of a connection addition attempt.
type ConnectionAttempt struct {
	SourceID    int
	TargetID    int
	Weight      float64
	ExactAcc    float64
	GenerousAcc float64
	ForgiveAcc  float64
	ModelJSON   string
	Improvement float64
}

// TryAddConnections attempts to improve accuracy by adding new random connections
// between neurons in a multithreaded manner. It tries up to maxAttempts to add
// connections that improve any of the accuracy metrics (exact, generous, forgiveness).
func (bp *Blueprint) TryAddConnections(
	sessions []Session,
	maxAttempts int,
) {
	fmt.Println("Starting TryAddConnections phase...")

	// Evaluate initial performance
	initialExact, initialGenerous, initialForgive, _, _, _ :=
		bp.EvaluateModelPerformance(sessions)

	var bestAttempt *ConnectionAttempt
	var bestImprovement float64
	var mu sync.Mutex // Mutex to protect access to bestAttempt and bestImprovement

	// Serialize the initial model
	initialModelJSON, err := bp.SerializeToJSON()
	if err != nil {
		fmt.Printf("Error serializing model: %v\n", err)
		return
	}

	// Channel to distribute unique connection pairs
	connectionCh := make(chan [2]int, maxAttempts)
	defer close(connectionCh)

	// Pre-generate unique connection pairs
	go func() {
		neuronIDs := bp.getAllNeuronIDs()
		rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })
		for i := 0; i < len(neuronIDs); i++ {
			for j := 0; j < len(neuronIDs); j++ {
				if i == j {
					continue
				}
				sourceID := neuronIDs[i]
				targetID := neuronIDs[j]
				if bp.connectionExists(sourceID, targetID) {
					continue
				}
				connectionCh <- [2]int{sourceID, targetID}
				if len(connectionCh) >= maxAttempts {
					return
				}
			}
		}
	}()

	// Determine the number of workers based on available CPU cores
	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}
	attemptsPerWorker := maxAttempts / numWorkers
	if attemptsPerWorker == 0 {
		attemptsPerWorker = 1
	}

	fmt.Printf("Launching %d worker(s) with up to %d attempts each.\n", numWorkers, attemptsPerWorker)

	// WaitGroup to wait for all workers to finish
	var wg sync.WaitGroup

	// Launch worker goroutines
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < attemptsPerWorker; i++ {
				connPair, ok := <-connectionCh
				if !ok {
					// No more connections to attempt
					return
				}
				sourceID, targetID := connPair[0], connPair[1]

				// Add a new connection with a random weight
				weight := rand.Float64()*2 - 1 // random weight between -1 and 1

				// Create a new Blueprint from the serialized model
				newBP := &Blueprint{}
				err := newBP.DeserializesFromJSON(initialModelJSON)
				if err != nil {
					fmt.Printf("Worker %d: Error deserializing model: %v\n", workerID, err)
					continue
				}

				// Add the connection
				err = newBP.addConnection(sourceID, targetID, weight)
				if err != nil {
					// Could not add connection, try again
					fmt.Printf("Worker %d: Error adding connection (%d -> %d): %v\n", workerID, sourceID, targetID, err)
					continue
				}

				// Evaluate the new model
				newExact, newGenerous, newForgive, _, _, _ := newBP.EvaluateModelPerformance(sessions)

				// Calculate improvement
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

				// Serialize the new model
				newModelJSON, err := newBP.SerializeToJSON()
				if err != nil {
					fmt.Printf("Worker %d: Error serializing new model: %v\n", workerID, err)
					continue
				}

				// If improvement, check if it's the best so far
				if improvement > bestImprovement {
					mu.Lock()
					if improvement > bestImprovement {
						bestImprovement = improvement
						bestAttempt = &ConnectionAttempt{
							SourceID:    sourceID,
							TargetID:    targetID,
							Weight:      weight,
							ExactAcc:    newExact,
							GenerousAcc: newGenerous,
							ForgiveAcc:  newForgive,
							ModelJSON:   newModelJSON,
							Improvement: improvement,
						}
					}
					mu.Unlock()
				}
			}
		}(w + 1)
	}

	// Wait for all workers to finish
	wg.Wait()

	// Apply the best improvement if any
	if bestAttempt != nil && bestAttempt.Improvement > 0 {
		// Deserialize the best model
		err := bp.DeserializesFromJSON(bestAttempt.ModelJSON)
		if err != nil {
			fmt.Printf("Error deserializing best model: %v\n", err)
			return
		}

		fmt.Printf("Added connection (%d -> %d) improved accuracy by %.6f!\n",
			bestAttempt.SourceID, bestAttempt.TargetID, bestAttempt.Improvement)
	} else {
		fmt.Println("No beneficial connections were found to improve the model.")
	}

	fmt.Println("TryAddConnections phase completed.")
}

// pickRandomNeuronsForConnection picks two neurons to connect, ensuring we do not form loops
// from output to output or connect a neuron to itself or duplicate an existing connection.
func (bp *Blueprint) pickRandomNeuronsForConnection() (int, int) {
	// This function is no longer needed as unique connection pairs are pre-generated.
	// It's kept here for backward compatibility or potential future use.
	return -1, -1
}

// connectionExists checks if a connection from source to target already exists.
func (bp *Blueprint) connectionExists(sourceID, targetID int) bool {
	targetNeuron, ok := bp.Neurons[targetID]
	if !ok {
		return false
	}
	for _, conn := range targetNeuron.Connections {
		if int(conn[0]) == sourceID {
			return true
		}
	}
	return false
}

// addConnection adds a connection from source to target with given weight.
func (bp *Blueprint) addConnection(sourceID, targetID int, weight float64) error {
	targetNeuron, ok := bp.Neurons[targetID]
	if !ok {
		return fmt.Errorf("target neuron %d does not exist", targetID)
	}

	// Add the connection
	targetNeuron.Connections = append(targetNeuron.Connections, []float64{float64(sourceID), weight})
	return nil
}

// removeConnection removes a connection from source to target.
func (bp *Blueprint) removeConnection(sourceID, targetID int) {
	targetNeuron, ok := bp.Neurons[targetID]
	if !ok {
		return
	}

	newConnections := [][]float64{}
	for _, conn := range targetNeuron.Connections {
		if int(conn[0]) != sourceID {
			newConnections = append(newConnections, conn)
		}
	}
	targetNeuron.Connections = newConnections
}
