// evaluation.go
package blueprint

import (
	//"fmt"
	"math"
)

// Session represents a training or testing session
type Session struct {
	InputVariables map[int]float64 // Inputs to the network (neuron ID to value)
	ExpectedOutput map[int]float64 // Expected outputs (neuron ID to value)
	Timesteps      int             // Number of timesteps to run (for recurrent networks)
}

// EvaluateModelPerformance evaluates the model's performance over a list of sessions,
// returning exact accuracy, generous accuracy, forgiveness accuracy, and their associated errors.
func (bp *Blueprint) EvaluateModelPerformance(sessions []Session, forgivenessThreshold float64) (float64, float64, float64, int, float64, int) {
	exactCorrectPredictions := 0
	totalGenerousScore := 0.0
	forgivenessCorrectPredictions := 0
	exactErrorCount := 0
	totalGenerousError := 0.0
	forgivenessErrorCount := 0

	for _, session := range sessions {
		bp.RunNetwork(session.InputVariables, session.Timesteps)
		predictedOutput := bp.GetOutputs()

		probs := softmaxMap(predictedOutput)
		predClass := argmaxMap(probs)
		expClass := argmaxMap(session.ExpectedOutput)

		if predClass == expClass {
			exactCorrectPredictions++
		} else {
			exactErrorCount++
		}

		similarityScore := calculateSimilarityScore(predictedOutput, session.ExpectedOutput)
		totalGenerousScore += similarityScore
		generousError := 100.0 - similarityScore
		totalGenerousError += generousError

		if isWithinForgivenessThreshold(predictedOutput, session.ExpectedOutput, forgivenessThreshold) {
			forgivenessCorrectPredictions++
		} else {
			forgivenessErrorCount++
		}
	}

	exactAccuracy := float64(exactCorrectPredictions) / float64(len(sessions)) * 100.0
	generousAccuracy := totalGenerousScore / float64(len(sessions))
	forgivenessAccuracy := float64(forgivenessCorrectPredictions) / float64(len(sessions)) * 100.0
	averageGenerousError := totalGenerousError / float64(len(sessions))

	return exactAccuracy, generousAccuracy, forgivenessAccuracy, exactErrorCount, averageGenerousError, forgivenessErrorCount
}

// Helper functions

// isPredictionExactCorrect checks if the model's predicted output matches the expected output within a small epsilon.
func isPredictionExactCorrect(predicted, expected map[int]float64) bool {
	const epsilon = 1e-6
	for id, expectedValue := range expected {
		if predictedValue, exists := predicted[id]; exists {
			if math.Abs(predictedValue-expectedValue) > epsilon {
				return false
			}
		} else {
			return false
		}
	}
	return true
}

// calculateSimilarityScore calculates a similarity score between predicted and expected outputs.
func calculateSimilarityScore(predicted, expected map[int]float64) float64 {
	if len(predicted) == 0 || len(expected) == 0 {
		return 0.0
	}

	totalAbsoluteError := 0.0
	for id, expectedValue := range expected {
		predictedValue, exists := predicted[id]
		if !exists {
			totalAbsoluteError += 1.0 // Assuming outputs are normalized between 0 and 1
			continue
		}
		difference := math.Abs(predictedValue - expectedValue)
		totalAbsoluteError += difference
	}

	// Calculate Mean Absolute Error (MAE)
	mae := totalAbsoluteError / float64(len(expected))

	// Convert MAE to similarity score
	similarity := (1.0 - mae) * 100.0

	// Ensure similarity is between 0 and 100
	if similarity < 0 {
		similarity = 0.0
	} else if similarity > 100.0 {
		similarity = 100.0
	}

	return similarity
}

// isWithinForgivenessThreshold checks if the predicted output is within the forgiveness threshold
// of the expected output for each output neuron.
func isWithinForgivenessThreshold(predicted, expected map[int]float64, threshold float64) bool {
	for id, expectedValue := range expected {
		predictedValue, exists := predicted[id]
		if !exists {
			return false
		}

		// Check if the predicted value is within the forgiveness threshold
		lowerBound := expectedValue * (1.0 - threshold)
		upperBound := expectedValue * (1.0 + threshold)

		if predictedValue < lowerBound || predictedValue > upperBound {
			return false
		}
	}

	return true
}
