package blueprint

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// TargetedMicroRefinement attempts to improve the model by focusing on "near-miss" samples
// and making small weight tweaks. It updates only if any accuracy improves without others decreasing.
func (bp *Blueprint) TargetedMicroRefinement(
	sessions []Session,
	maxIterations int,
	sampleSubsetSize int,
	connectionTrialsPerSample int,
	improvementThreshold float64,
) {
	rand.Seed(time.Now().UnixNano())

	exactAcc, generousAcc, forgiveAcc, _, _, _ := bp.EvaluateModelPerformance(sessions)
	fmt.Printf("Starting TargetedMicroRefinement: Exact=%.6f%%, Generous=%.6f%%, Forgiveness=%.6f%%\n",
		exactAcc, generousAcc, forgiveAcc)

	if exactAcc > improvementThreshold {
		fmt.Println("Already beyond improvement threshold. No refinement needed.")
		return
	}

	// Find near-miss samples at 80% generous cutoff
	nearMissSamples := bp.findNearMissSamples(sessions, 0.8)
	if len(nearMissSamples) == 0 {
		fmt.Println("No near-miss samples found at 80% generous cutoff. Trying 50% cutoff...")
		nearMissSamples = bp.findNearMissSamples(sessions, 0.5)
		if len(nearMissSamples) == 0 {
			fmt.Println("No near-miss samples found even at 50% cutoff. Nothing to refine.")
			return
		}
	}

	noImprovementCount := 0
	lastExactAcc := exactAcc
	lastGenerousAcc := generousAcc
	lastForgiveAcc := forgiveAcc

	for iter := 1; iter <= maxIterations; iter++ {
		fmt.Printf("--- Refine Iteration %d ---\n", iter)

		subset := sampleSubset(nearMissSamples, sampleSubsetSize)
		for _, s := range subset {
			criticalConnections := bp.identifyCriticalConnections()
			_ = bp.refineSampleWeights(s, criticalConnections, connectionTrialsPerSample)
		}

		newExactAcc, newGenerousAcc, newForgiveAcc, _, _, _ :=
			bp.EvaluateModelPerformance(sessions)

		fmt.Printf("After iteration %d:\n", iter)
		fmt.Printf("Exact=%.6f%% (was %.6f%%), Generous=%.6f%% (was %.6f%%), Forgiveness=%.6f%% (was %.6f%%)\n",
			newExactAcc, lastExactAcc, newGenerousAcc, lastGenerousAcc, newForgiveAcc, lastForgiveAcc)

		// Check for improvement without regression
		improvement := false
		if newExactAcc >= lastExactAcc && newGenerousAcc >= lastGenerousAcc && newForgiveAcc >= lastForgiveAcc {
			if newExactAcc > lastExactAcc {
				fmt.Println("Exact accuracy improved!")
				improvement = true
			}
			if newGenerousAcc > lastGenerousAcc {
				fmt.Println("Generous accuracy improved!")
				improvement = true
			}
			if newForgiveAcc > lastForgiveAcc {
				fmt.Println("Forgiveness accuracy improved!")
				improvement = true
			}
		}

		if improvement {
			lastExactAcc = newExactAcc
			lastGenerousAcc = newGenerousAcc
			lastForgiveAcc = newForgiveAcc
			noImprovementCount = 0
		} else {
			noImprovementCount++
			fmt.Printf("No improvement in metrics this iteration. Count=%d\n", noImprovementCount)
		}

		if newExactAcc >= improvementThreshold {
			fmt.Printf("Reached improvement threshold of %.6f%% exact accuracy.\n", improvementThreshold)
			break
		}

		if noImprovementCount > 5 {
			fmt.Println("No improvement in several iterations. Stopping refinement.")
			break
		}
	}
}

// findNearMissSamples identifies sessions where the network is close but not exact based on generousCutoff.
func (bp *Blueprint) findNearMissSamples(sessions []Session, generousCutoff float64) []Session {
	var nearMiss []Session
	countChecked := 0
	countQualified := 0

	for _, s := range sessions {
		bp.RunNetwork(s.InputVariables, s.Timesteps)
		output := bp.GetOutputs()

		if isPredictionExactCorrect(output, s.ExpectedOutput) {
			continue
		}

		similarity := calculateSimilarityScore(output, s.ExpectedOutput)
		countChecked++
		if similarity >= generousCutoff*100.0 {
			nearMiss = append(nearMiss, s)
			countQualified++
		}
	}

	fmt.Printf("Checked %d samples for near-miss at cutoff=%.2f%%, found %d qualifying.\n",
		countChecked, generousCutoff*100.0, countQualified)
	return nearMiss
}

// sampleSubset selects up to n random samples
func sampleSubset(sessions []Session, n int) []Session {
	if len(sessions) <= n {
		return sessions
	}
	rand.Shuffle(len(sessions), func(i, j int) { sessions[i], sessions[j] = sessions[j], sessions[i] })
	return sessions[:n]
}

// identifyCriticalConnections returns output neurons by default.
// You can enhance this method to identify other key neurons if needed.
func (bp *Blueprint) identifyCriticalConnections() []int {
	return bp.OutputNodes
}

// refineSampleWeights tries small perturbations on weights for one sample.
func (bp *Blueprint) refineSampleWeights(
	sample Session,
	criticalNeurons []int,
	trials int,
) bool {
	initialError := bp.sampleError(sample)
	improved := false

	if len(criticalNeurons) == 0 {
		fmt.Println("No critical neurons identified. Skipping this sample.")
		return false
	}

	for trial := 0; trial < trials; trial++ {
		nID := criticalNeurons[rand.Intn(len(criticalNeurons))]
		neuron, ok := bp.Neurons[nID]
		if !ok || len(neuron.Connections) == 0 {
			continue
		}

		cIndex := rand.Intn(len(neuron.Connections))
		oldWeight := neuron.Connections[cIndex][1]
		delta := rand.NormFloat64() * 0.01

		// Try positive delta
		neuron.Connections[cIndex][1] = oldWeight + delta
		newError := bp.sampleError(sample)
		if newError < initialError {
			initialError = newError
			improved = true
			fmt.Printf("Improved sample error with +delta=%.6f on connection %d of neuron %d\n", delta, cIndex, nID)
		} else {
			// revert and try negative delta
			neuron.Connections[cIndex][1] = oldWeight - delta
			newError = bp.sampleError(sample)
			if newError < initialError {
				initialError = newError
				improved = true
				fmt.Printf("Improved sample error with -delta=%.6f on connection %d of neuron %d\n", delta, cIndex, nID)
			} else {
				// revert to original if no improvement
				neuron.Connections[cIndex][1] = oldWeight
			}
		}
	}

	if !improved {
		fmt.Println("No improvements made on this sample after all trials.")
	}
	return improved
}

// sampleError computes MAE for a single sample
func (bp *Blueprint) sampleError(sample Session) float64 {
	bp.RunNetwork(sample.InputVariables, sample.Timesteps)
	output := bp.GetOutputs()
	return sampleMAE(output, sample.ExpectedOutput)
}

// sampleMAE calculates mean absolute error for predicted vs. expected
func sampleMAE(predicted, expected map[int]float64) float64 {
	if len(expected) == 0 {
		return 0.0
	}
	totalError := 0.0
	for id, expVal := range expected {
		predVal, exists := predicted[id]
		if !exists {
			totalError += 1.0
			continue
		}
		totalError += math.Abs(predVal - expVal)
	}
	return totalError / float64(len(expected))
}
