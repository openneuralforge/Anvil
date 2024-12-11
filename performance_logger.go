// blueprint/performance_logger.go
package blueprint

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// SessionPerformance holds performance metrics for a single session.
type SessionPerformance struct {
	SessionID            int
	ExactAccuracy        float64
	GenerousAccuracy     float64
	ForgiveAccuracy      float64
	ErrorMetric          float64
	Timestamp            string
	PredictedClass       int
	ExpectedClass        int
	PredictedProbability float64
}

// PerformanceLogger handles logging of session performances.
type PerformanceLogger struct {
	LogDir   string
	FilePath string
	mu       sync.Mutex
}

// NewPerformanceLogger initializes a new PerformanceLogger.
// logDir specifies the directory where logs will be saved.
func NewPerformanceLogger(logDir string) (*PerformanceLogger, error) {
	// Create the log directory if it doesn't exist
	if err := os.MkdirAll(logDir, os.ModePerm); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %v", err)
	}

	// Create a timestamped CSV file
	timestamp := time.Now().Format("20060102_150405")
	fileName := fmt.Sprintf("performance_log_%s.csv", timestamp)
	filePath := filepath.Join(logDir, fileName)

	// Create the CSV file and write headers
	file, err := os.Create(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header row
	header := []string{
		"SessionID",
		"ExactAccuracy",
		"GenerousAccuracy",
		"ForgiveAccuracy",
		"ErrorMetric",
		"PredictedClass",
		"ExpectedClass",
		"PredictedProbability",
		"Timestamp",
	}
	if err := writer.Write(header); err != nil {
		return nil, fmt.Errorf("failed to write header to CSV: %v", err)
	}

	return &PerformanceLogger{
		LogDir:   logDir,
		FilePath: filePath,
	}, nil
}

// Log appends a SessionPerformance record to the CSV file.
func (pl *PerformanceLogger) Log(sp SessionPerformance) error {
	pl.mu.Lock()
	defer pl.mu.Unlock()

	// Open the CSV file in append mode
	file, err := os.OpenFile(pl.FilePath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open CSV file for appending: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Prepare the data row
	row := []string{
		fmt.Sprintf("%d", sp.SessionID),
		fmt.Sprintf("%.4f", sp.ExactAccuracy),
		fmt.Sprintf("%.4f", sp.GenerousAccuracy),
		fmt.Sprintf("%.4f", sp.ForgiveAccuracy),
		fmt.Sprintf("%.4f", sp.ErrorMetric),
		fmt.Sprintf("%d", sp.PredictedClass),
		fmt.Sprintf("%d", sp.ExpectedClass),
		fmt.Sprintf("%.4f", sp.PredictedProbability),
		sp.Timestamp,
	}

	// Write the data row
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write row to CSV: %v", err)
	}

	return nil
}

// EvaluateAndLogPerformance evaluates each session and logs the performance metrics.
// This function runs independently of training processes.
// You must pass the sessions you want to evaluate.
func (bp *Blueprint) EvaluateAndLogPerformance(sessions []Session, logger *PerformanceLogger) error {
	var wg sync.WaitGroup
	metricsCh := make(chan SessionPerformance, len(sessions))
	errorCh := make(chan error, len(sessions))

	for idx, session := range sessions {
		wg.Add(1)
		go func(sessionID int, sess Session) {
			defer wg.Done()

			bp.RunNetwork(sess.InputVariables, sess.Timesteps)
			predictedOutput := bp.GetOutputs()

			// Determine predicted class and its probability
			probs := softmaxMap(predictedOutput)
			predClass, predProb := argmaxWithProb(probs)
			expClass := argmaxMap(sess.ExpectedOutput)

			// Calculate metrics
			exactAcc, generousAcc, forgiveAcc := calculateAccuracies(predClass, expClass)
			errorMetric := 100.0 - exactAcc

			metricsCh <- SessionPerformance{
				SessionID:            sessionID,
				ExactAccuracy:        exactAcc,
				GenerousAccuracy:     generousAcc,
				ForgiveAccuracy:      forgiveAcc,
				ErrorMetric:          errorMetric,
				PredictedClass:       predClass,
				ExpectedClass:        expClass,
				PredictedProbability: predProb,
				Timestamp:            time.Now().Format(time.RFC3339),
			}
		}(idx+1, session)
	}

	wg.Wait()
	close(metricsCh)
	close(errorCh)

	if len(errorCh) > 0 {
		return fmt.Errorf("errors occurred during evaluation")
	}

	for sp := range metricsCh {
		if err := logger.Log(sp); err != nil {
			return fmt.Errorf("failed to log performance for session %d: %v", sp.SessionID, err)
		}
	}
	return nil
}

// calculateAccuracies computes Exact, Generous, and Forgive accuracies based on prediction.
func calculateAccuracies(predClass, expClass int) (exactAcc, generousAcc, forgiveAcc float64) {
	if predClass == expClass {
		exactAcc = 100.0
		generousAcc = 100.0
		forgiveAcc = 100.0
	} else {
		exactAcc = 0.0
		generousAcc = 50.0 // Example: partial credit
		forgiveAcc = 25.0  // Example: minimal credit
	}
	return
}

// softmaxMap applies softmax to the values in a map and returns a new map with probabilities.
func softmaxMap(m map[int]float64) map[int]float64 {
	var sumExp float64
	for _, v := range m {
		sumExp += math.Exp(v)
	}
	probs := make(map[int]float64)
	for k, v := range m {
		probs[k] = math.Exp(v) / sumExp
	}
	return probs
}

// argmaxMap returns the key of the maximum value in the map.
// Assumes that the map is non-empty.
func argmaxMap(m map[int]float64) int {
	var maxKey int
	var maxVal float64 = -math.MaxFloat64
	for k, v := range m {
		if v > maxVal {
			maxVal = v
			maxKey = k
		}
	}
	return maxKey // Directly return the key as the class index
}

// argmaxWithProb returns the key of the maximum value in the map and its probability.
// Assumes that the map is non-empty.
func argmaxWithProb(m map[int]float64) (int, float64) {
	var maxKey int
	var maxVal float64 = -math.MaxFloat64
	for k, v := range m {
		if v > maxVal {
			maxVal = v
			maxKey = k
		}
	}
	return maxKey, maxVal
}
