// evaluation.go
package blueprint

import (
	"math"
)

// Session represents a training or testing session
type Session struct {
	InputVariables map[int]float64 // Inputs to the network (neuron ID to value)
	ExpectedOutput map[int]float64 // Expected outputs (neuron ID to value)
	Timesteps      int             // Number of timesteps to run (for recurrent networks)
}

// EvaluateModelPerformance evaluates the model's performance over a list of sessions,
// returning exact accuracy, generous accuracy, decile consistency accuracy, and their associated errors.
func (bp *Blueprint) EvaluateModelPerformance(sessions []Session) (float64, float64, float64, int, float64, int) {
	exactCorrectPredictions := 0
	decileConsistentCount := 0
	exactErrorCount := 0
	totalGenerousValue := 0.0
	totalGenerousError := 0.0
	decileInconsistentCount := 0

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

		generousValue := calculateGenerousValue(predictedOutput, session.ExpectedOutput)
		totalGenerousValue += generousValue
		generousError := getMaxFloat() - generousValue
		totalGenerousError += generousError

		if isDecileConsistent(predictedOutput, session.ExpectedOutput) {
			decileConsistentCount++
		} else {
			decileInconsistentCount++
		}
	}

	exactAccuracy := float64(exactCorrectPredictions) / float64(len(sessions)) * 100.0
	generousAccuracy := totalGenerousValue / float64(len(sessions))
	decileConsistencyAccuracy := float64(decileConsistentCount) / float64(len(sessions)) * 100.0
	averageGenerousError := totalGenerousError / float64(len(sessions))

	return exactAccuracy, generousAccuracy, decileConsistencyAccuracy, exactErrorCount, averageGenerousError, decileInconsistentCount
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

// isDecileConsistent checks if the predicted output falls consistently within the same decile across all expected values.
func isDecileConsistent(predicted, expected map[int]float64) bool {
	const decileStep = 0.1 // Deciles: 10%, 20%, ..., 90%

	var referenceDecile int
	for id, expectedValue := range expected {
		predictedValue, exists := predicted[id]
		if !exists {
			return false
		}

		// Determine the decile for the current predicted value
		difference := math.Abs(predictedValue - expectedValue)
		decile := int(difference / decileStep)
		if decile > 9 {
			decile = 9 // Cap at the 90% decile
		}

		if referenceDecile == 0 {
			referenceDecile = decile
		} else if referenceDecile != decile {
			return false
		}
	}

	return true
}

// calculateGenerousValue calculates a similarity value based on the highest floating-point value.
func calculateGenerousValue(predicted, expected map[int]float64) float64 {
	if len(predicted) == 0 || len(expected) == 0 {
		return 0.0
	}

	totalDifference := 0.0
	count := 0
	for id, expectedValue := range expected {
		predictedValue, exists := predicted[id]
		if !exists {
			continue
		}
		difference := math.Abs(predictedValue - expectedValue)
		totalDifference += difference
		count++
	}

	if count == 0 {
		return 0.0 // Avoid division by zero
	}

	meanDifference := totalDifference / float64(count)
	generousValue := 1.0 - meanDifference // Ensure the value is bounded between 0 and 1

	if generousValue < 0 {
		generousValue = 0.0
	}

	return generousValue
}

// calculateClassSensitivity penalizes misclassifications more heavily for certain classes.
func calculateClassSensitivity(predicted, expected map[int]float64) float64 {
	penaltyFactor := 2.0 // Example sensitivity multiplier
	totalPenalty := 0.0
	for id, expectedValue := range expected {
		predictedValue, exists := predicted[id]
		if !exists {
			totalPenalty += penaltyFactor
			continue
		}
		difference := math.Abs(predictedValue - expectedValue)
		if expectedValue > 0.8 { // Example condition for "sensitive" classes
			totalPenalty += difference * penaltyFactor
		} else {
			totalPenalty += difference
		}
	}
	return getMaxFloat() - totalPenalty
}

// calculateWeightedProximity evaluates the closeness of predictions to expected values, weighted by their importance.
func calculateWeightedProximity(predicted, expected map[int]float64) float64 {
	if len(predicted) == 0 || len(expected) == 0 {
		return 0.0
	}

	totalWeightedDifference := 0.0
	totalWeight := 0.0
	for id, expectedValue := range expected {
		predictedValue, exists := predicted[id]
		if !exists {
			continue
		}
		difference := math.Abs(predictedValue - expectedValue)
		weight := math.Max(expectedValue, 0.1) // Avoid zero weight
		totalWeightedDifference += difference * weight
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0.0 // Avoid division by zero
	}

	return 1.0 - (totalWeightedDifference / totalWeight) // Ensure proximity is normalized between 0 and 1
}

func (bp *Blueprint) AdvancedEvaluateModelPerformance(sessions []Session) (float64, float64, map[string]float64, float64, int, float64, int) {
	exactCorrectPredictions := 0
	totalGenerousValue := 0.0
	totalAdvancedMetrics := map[string]float64{
		"weightedProximity":   0.0,
		"classSensitivity":    0.0,
		"temporalConsistency": 0.0,
	}
	totalGenerousError := 0.0
	exactErrorCount := 0
	decileConsistentCount := 0
	decileInconsistentCount := 0

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

		// Old generous accuracy
		generousValue := calculateGenerousValue(predictedOutput, session.ExpectedOutput)
		totalGenerousValue += generousValue
		generousError := getMaxFloat() - generousValue
		totalGenerousError += generousError

		// Advanced generous metrics
		totalAdvancedMetrics["weightedProximity"] += calculateWeightedProximity(predictedOutput, session.ExpectedOutput)
		totalAdvancedMetrics["classSensitivity"] += calculateClassSensitivity(predictedOutput, session.ExpectedOutput)
		totalAdvancedMetrics["temporalConsistency"] += calculateTemporalConsistency(predictedOutput, session.ExpectedOutput)

		if isDecileConsistent(predictedOutput, session.ExpectedOutput) {
			decileConsistentCount++
		} else {
			decileInconsistentCount++
		}
	}

	exactAccuracy := float64(exactCorrectPredictions) / float64(len(sessions)) * 100.0
	generousAccuracy := totalGenerousValue / float64(len(sessions))
	for metric := range totalAdvancedMetrics {
		totalAdvancedMetrics[metric] /= float64(len(sessions))
	}
	averageGenerousError := totalGenerousError / float64(len(sessions))
	decileConsistencyAccuracy := float64(decileConsistentCount) / float64(len(sessions)) * 100.0

	return exactAccuracy, generousAccuracy, totalAdvancedMetrics, decileConsistencyAccuracy, exactErrorCount, averageGenerousError, decileInconsistentCount
}

// calculateTemporalConsistency evaluates stability across timesteps for recurrent networks.
func calculateTemporalConsistency(predicted, expected map[int]float64) float64 {
	if len(predicted) == 0 || len(expected) == 0 {
		return 0.0
	}

	// Assuming predicted and expected contain temporal sequences as time-indexed keys.
	// You can replace this logic with the actual representation of your time-indexed data.

	var prevPredicted float64
	var prevExpected float64
	temporalStability := 0.0
	numComparisons := 0

	for t := 1; t < len(predicted); t++ { // Assuming keys are time indices
		currentPredicted := predicted[t]
		currentExpected := expected[t]

		if t > 1 { // Compare with previous values to assess consistency
			predictedDelta := math.Abs(currentPredicted - prevPredicted)
			expectedDelta := math.Abs(currentExpected - prevExpected)

			// Penalize large inconsistencies
			stabilityScore := math.Max(0, 1-math.Abs(predictedDelta-expectedDelta))
			temporalStability += stabilityScore
			numComparisons++
		}

		prevPredicted = currentPredicted
		prevExpected = currentExpected
	}

	if numComparisons == 0 {
		return 0.0
	}

	// Normalize stability score
	return temporalStability / float64(numComparisons) * 100.0
}
