// nas.go
package blueprint

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"
	"bytes"
	"runtime"
	"sync"
)

// This struct stores the result of evaluating a candidate blueprint.
type candidateResult struct {
	ExactAccuracy       float64
	GenerousAccuracy    float64
	ForgivenessAccuracy float64
	CandidateBlueprint  *Blueprint
}

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

// SimpleNASWithoutCrossover performs a basic neural architecture search by incrementally adding one neuron at a time
// and keeping the change if it improves the model's evaluation on any of the specified evaluation metrics.
func (bp *Blueprint) SimpleNASWithoutCrossover(
	sessions []Session,
	maxIterations int,
	forgivenessThreshold float64,
	neuronTypes []string,
	metricsToOptimize []string,
) {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Validate and normalize metricsToOptimize
	validMetrics := map[string]bool{
		"exact":       true,
		"generous":    true,
		"forgiveness": true,
	}
	selectedMetrics := make(map[string]bool)
	for _, metric := range metricsToOptimize {
		metricLower := strings.ToLower(metric)
		if _, exists := validMetrics[metricLower]; exists {
			selectedMetrics[metricLower] = true
		} else {
			fmt.Printf("Warning: Invalid metric '%s' ignored.\n", metric)
		}
	}

	if len(selectedMetrics) == 0 {
		fmt.Println("No valid metrics specified for optimization. Exiting NAS.")
		return
	}

	// Evaluate the initial model
	initialExact, initialGenerous, initialForgiveness, _, _, _ := bp.EvaluateModelPerformance(sessions, forgivenessThreshold)
	fmt.Printf("Initial model performance: Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
		initialExact, initialGenerous, initialForgiveness)

	// Initialize best metrics based on selectedMetrics
	bestExact, bestGenerous, bestForgiveness := initialExact, initialGenerous, initialForgiveness

	for iteration := 1; iteration <= maxIterations; iteration++ {
		fmt.Printf("Iteration %d\n", iteration)

		// Clone the current blueprint
		candidateBlueprint := bp.Clone()
		if candidateBlueprint == nil {
			fmt.Println("Cloning failed. Skipping iteration.")
			continue
		}

		// Randomly select a neuron type to insert
		neuronType := neuronTypes[rand.Intn(len(neuronTypes))]

		// Insert a neuron of the selected type
		err := candidateBlueprint.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
		if err != nil {
			fmt.Printf("Iteration %d: Failed to insert neuron of type '%s': %v\n", iteration, neuronType, err)
			continue
		}

		// Evaluate the candidate model
		exactAcc, generousAcc, forgivenessAcc, _, _, _ := candidateBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

		// Determine if there's an improvement based on selected metrics
		improved := false
		for metric := range selectedMetrics {
			switch metric {
			case "exact":
				if exactAcc > bestExact {
					improved = true
				}
			case "generous":
				if generousAcc > bestGenerous {
					improved = true
				}
			case "forgiveness":
				if forgivenessAcc > bestForgiveness {
					improved = true
				}
			}
		}

		if improved {
			// Update the best model
			*bp = *candidateBlueprint
			if selectedMetrics["exact"] && exactAcc > bestExact {
				bestExact = exactAcc
			}
			if selectedMetrics["generous"] && generousAcc > bestGenerous {
				bestGenerous = generousAcc
			}
			if selectedMetrics["forgiveness"] && forgivenessAcc > bestForgiveness {
				bestForgiveness = forgivenessAcc
			}

			// Log the improvement
			improvementLog := "Iteration %d: Improved model found! "
			args := []interface{}{iteration}
			if selectedMetrics["exact"] {
				improvementLog += "Exact=%.2f%%, "
				args = append(args, exactAcc)
			}
			if selectedMetrics["generous"] {
				improvementLog += "Generous=%.2f%%, "
				args = append(args, generousAcc)
			}
			if selectedMetrics["forgiveness"] {
				improvementLog += "Forgiveness=%.2f%%, "
				args = append(args, forgivenessAcc)
			}
			// Remove trailing comma and space
			improvementLog = strings.TrimSuffix(improvementLog, ", ")
			fmt.Printf(improvementLog+"\n", args...)
		} else {
			fmt.Printf("Iteration %d: No improvement.\n", iteration)
		}

		// Early stopping if any selected metric reaches 100%
		perfect := false
		for metric := range selectedMetrics {
			switch metric {
			case "exact":
				if bestExact == 100.0 {
					perfect = true
				}
			case "generous":
				if bestGenerous == 100.0 {
					perfect = true
				}
			case "forgiveness":
				if bestForgiveness == 100.0 {
					perfect = true
				}
			}
		}
		if perfect {
			fmt.Println("Perfect accuracy achieved on one of the selected metrics. Stopping NAS.")
			break
		}
	}

	fmt.Println("SimpleNASWithoutCrossover training completed.")
}

// SimpleNASWithRandomConnections incrementally adds neurons with random connections,
// performs hill-climbing weight updates, and stores the evaluation progress.
// It ensures only architectures with better or equal exact accuracy and improved generous or forgiveness accuracy are accepted.
func (bp *Blueprint) SimpleNASWithRandomConnections(
	sessions []Session,
	maxIterations int,
	forgivenessThreshold float64,
	neuronTypes []string,
	weightUpdateIterations int, // Number of hill-climbing steps per NAS iteration
) {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Keep track of the best model and its performance
	bestBlueprint := bp.Clone() // Assume we have a Clone method
	if bestBlueprint == nil {
		fmt.Println("Failed to clone the initial blueprint.")
		return
	}

	bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy, _, _, _ := bestBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

	// Array to store progress
	progress := []struct {
		Iteration           int
		ExactAccuracy       float64
		GenerousAccuracy    float64
		ForgivenessAccuracy float64
	}{
		{
			Iteration:           0,
			ExactAccuracy:       bestExactAccuracy,
			GenerousAccuracy:    bestGenerousAccuracy,
			ForgivenessAccuracy: bestForgivenessAccuracy,
		},
	}

	fmt.Printf("Initial model performance: Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
		bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy)

	for iteration := 1; iteration <= maxIterations; iteration++ {
		fmt.Printf("=== Iteration %d ===\n", iteration)

		// Clone the best blueprint to create a new candidate
		candidateBlueprint := bestBlueprint.Clone()
		if candidateBlueprint == nil {
			fmt.Printf("Iteration %d: Failed to clone the best blueprint.\n", iteration)
			continue
		}

		// Randomly select a neuron type to add
		neuronType := neuronTypes[rand.Intn(len(neuronTypes))]

		// Insert a neuron of this type between inputs and outputs
		err := candidateBlueprint.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
		if err != nil {
			fmt.Printf("Iteration %d: Failed to insert neuron of type '%s': %v\n", iteration, neuronType, err)
			continue
		}

		// Perform hill-climbing weight updates
		for w := 0; w < weightUpdateIterations; w++ {
			improved := candidateBlueprint.HillClimbWeightUpdate(sessions, forgivenessThreshold)
			if !improved {
				// If no improvement, you can choose to continue or break early
				// Here, we'll continue for the specified number of iterations
				continue
			}
		}

		// Evaluate the candidate model after weight updates
		exactAccuracy, generousAccuracy, forgivenessAccuracy, _, _, _ := candidateBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

		// Check if the candidate model improves on any of the three metrics
		if exactAccuracy > bestExactAccuracy ||
			(exactAccuracy == bestExactAccuracy && (generousAccuracy > bestGenerousAccuracy || forgivenessAccuracy > bestForgivenessAccuracy)) {
			// Update the best model
			bestBlueprint = candidateBlueprint
			bestExactAccuracy = exactAccuracy
			bestGenerousAccuracy = generousAccuracy
			bestForgivenessAccuracy = forgivenessAccuracy

			fmt.Printf("Iteration %d: Improved model found! Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
				iteration, exactAccuracy, generousAccuracy, forgivenessAccuracy)

			// Store progress
			progress = append(progress, struct {
				Iteration           int
				ExactAccuracy       float64
				GenerousAccuracy    float64
				ForgivenessAccuracy float64
			}{
				Iteration:           iteration,
				ExactAccuracy:       bestExactAccuracy,
				GenerousAccuracy:    bestGenerousAccuracy,
				ForgivenessAccuracy: bestForgivenessAccuracy,
			})
		} else {
			fmt.Printf("Iteration %d: No improvement.\n", iteration)
		}

		// Early stopping if exact accuracy reaches 100%
		if bestExactAccuracy == 100.0 {
			fmt.Println("Perfect exact accuracy achieved. Stopping NAS.")
			break
		}
	}

	// Print progress
	fmt.Println("NAS Progress:")
	for _, record := range progress {
		fmt.Printf("Iteration %d: Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
			record.Iteration, record.ExactAccuracy, record.GenerousAccuracy, record.ForgivenessAccuracy)
	}

	// Update the original blueprint with the best found
	*bp = *bestBlueprint
}

// getRandomXNeurons retrieves `x` random neurons from the list, or fewer if not enough exist.
func getRandomXNeurons(neuronIDs []int, x int) []int {
	if len(neuronIDs) <= x {
		return neuronIDs
	}
	rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })
	return neuronIDs[:x]
}

// ParallelSimpleNASWithRandomConnections attempts to improve the blueprint using multi-threading.
// It automatically detects the number of CPU cores and runs multiple candidate tests per iteration.
func (bp *Blueprint) ParallelSimpleNASWithRandomConnections(
	sessions []Session,
	maxIterations int,
	forgivenessThreshold float64,
	neuronTypes []string,
	weightUpdateIterations int,
) {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Clone the initial blueprint
	bestBlueprint := bp.Clone()
	if bestBlueprint == nil {
		fmt.Println("Failed to clone the initial blueprint.")
		return
	}

	bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy, _, _, _ :=
		bestBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

	progress := []struct {
		Iteration           int
		ExactAccuracy       float64
		GenerousAccuracy    float64
		ForgivenessAccuracy float64
	}{
		{
			Iteration:           0,
			ExactAccuracy:       bestExactAccuracy,
			GenerousAccuracy:    bestGenerousAccuracy,
			ForgivenessAccuracy: bestForgivenessAccuracy,
		},
	}

	fmt.Printf("Initial model performance: Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
		bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy)

	// Determine the level of parallelism based on the number of CPU cores
	numWorkers := runtime.NumCPU()
	fmt.Printf("Running with %d parallel workers.\n", numWorkers)

	// Convert the best blueprint to a JSON string for distribution to workers
	serializeBlueprint := func(bp *Blueprint) (string, error) {
		data, err := json.Marshal(bp)
		if err != nil {
			return "", err
		}
		return string(data), nil
	}

	deserializeBlueprint := func(data string) (*Blueprint, error) {
		var newBP Blueprint
		dec := json.NewDecoder(bytes.NewReader([]byte(data)))
		if err := dec.Decode(&newBP); err != nil {
			return nil, err
		}
		// Re-initialize any maps or activation functions that are not serialized
		newBP.InitializeActivationFunctions()
		return &newBP, nil
	}

	for iteration := 1; iteration <= maxIterations; iteration++ {
		fmt.Printf("=== Iteration %d ===\n", iteration)

		// Serialize the current best blueprint
		bestBPJSON, err := serializeBlueprint(bestBlueprint)
		if err != nil {
			fmt.Printf("Iteration %d: Failed to serialize blueprint: %v\n", iteration, err)
			continue
		}

		// We'll generate multiple candidates in parallel and pick the best
		var wg sync.WaitGroup
		resultsChan := make(chan candidateResult, numWorkers)

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Deserialize the blueprint for this worker
				candidateBlueprint, err := deserializeBlueprint(bestBPJSON)
				if err != nil {
					fmt.Printf("Worker: Failed to deserialize blueprint: %v\n", err)
					return
				}

				// Randomly select a neuron type to add
				neuronType := neuronTypes[rand.Intn(len(neuronTypes))]

				// Insert a neuron of this type between inputs and outputs
				err = candidateBlueprint.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
				if err != nil {
					// Failed to insert a neuron, no result from this worker
					return
				}

				// Perform hill-climbing weight updates
				for w := 0; w < weightUpdateIterations; w++ {
					improved := candidateBlueprint.HillClimbWeightUpdate(sessions, forgivenessThreshold)
					if !improved {
						continue
					}
				}

				// Evaluate the candidate model after weight updates
				exactAccuracy, generousAccuracy, forgivenessAccuracy, _, _, _ :=
					candidateBlueprint.EvaluateModelPerformance(sessions, forgivenessThreshold)

				resultsChan <- candidateResult{
					ExactAccuracy:       exactAccuracy,
					GenerousAccuracy:    generousAccuracy,
					ForgivenessAccuracy: forgivenessAccuracy,
					CandidateBlueprint:  candidateBlueprint,
				}
			}()
		}

		// Wait for all workers to finish
		wg.Wait()
		close(resultsChan)

		// Find the best candidate from this iteration
		improved := false
		var bestIterationCandidate *Blueprint
		var bestItExact, bestItGenerous, bestItForgiveness float64

		// Initially set improvement metrics to current best for comparison
		bestItExact = bestExactAccuracy
		bestItGenerous = bestGenerousAccuracy
		bestItForgiveness = bestForgivenessAccuracy

		for res := range resultsChan {
			// Check if the candidate model improves on any of the metrics according to our criteria
			if res.ExactAccuracy > bestExactAccuracy ||
				(res.ExactAccuracy == bestExactAccuracy && (res.GenerousAccuracy > bestGenerousAccuracy || res.ForgivenessAccuracy > bestForgivenessAccuracy)) {
				if res.ExactAccuracy > bestItExact ||
					(res.ExactAccuracy == bestItExact && (res.GenerousAccuracy > bestItGenerous || res.ForgivenessAccuracy > bestItForgiveness)) {
					bestIterationCandidate = res.CandidateBlueprint
					bestItExact = res.ExactAccuracy
					bestItGenerous = res.GenerousAccuracy
					bestItForgiveness = res.ForgivenessAccuracy
					improved = true
				}
			}
		}

		if improved && bestIterationCandidate != nil {
			bestBlueprint = bestIterationCandidate
			bestExactAccuracy = bestItExact
			bestGenerousAccuracy = bestItGenerous
			bestForgivenessAccuracy = bestItForgiveness

			fmt.Printf("Iteration %d: Improved model found! Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
				iteration, bestExactAccuracy, bestGenerousAccuracy, bestForgivenessAccuracy)

			progress = append(progress, struct {
				Iteration           int
				ExactAccuracy       float64
				GenerousAccuracy    float64
				ForgivenessAccuracy float64
			}{
				Iteration:           iteration,
				ExactAccuracy:       bestExactAccuracy,
				GenerousAccuracy:    bestGenerousAccuracy,
				ForgivenessAccuracy: bestForgivenessAccuracy,
			})
		} else {
			fmt.Printf("Iteration %d: No improvement.\n", iteration)
		}

		// Early stopping if exact accuracy reaches 100%
		if bestExactAccuracy == 100.0 {
			fmt.Println("Perfect exact accuracy achieved. Stopping NAS.")
			break
		}
	}

	// Print progress
	fmt.Println("NAS Progress:")
	for _, record := range progress {
		fmt.Printf("Iteration %d: Exact=%.2f%%, Generous=%.2f%%, Forgiveness=%.2f%%\n",
			record.Iteration, record.ExactAccuracy, record.GenerousAccuracy, record.ForgivenessAccuracy)
	}

	// Update the original blueprint with the best found
	*bp = *bestBlueprint
}