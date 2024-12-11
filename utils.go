package blueprint

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
)

// Softmax activation function (applied across a slice)
func Softmax(inputs []float64) []float64 {
	max := inputs[0]
	for _, v := range inputs {
		if v > max {
			max = v
		}
	}
	expSum := 0.0
	expInputs := make([]float64, len(inputs))
	for i, v := range inputs {
		expInputs[i] = math.Exp(v - max)
		expSum += expInputs[i]
	}
	for i := range expInputs {
		expInputs[i] /= expSum
	}
	return expInputs
}

// LoadNeurons loads neurons from a JSON string
func (bp *Blueprint) LoadNeurons(jsonData string) error {

	var rawNeurons []json.RawMessage
	if err := json.Unmarshal([]byte(jsonData), &rawNeurons); err != nil {
		return err
	}

	for _, rawNeuron := range rawNeurons {
		var baseNeuron struct {
			ID   int    `json:"id"`
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rawNeuron, &baseNeuron); err != nil {
			return err
		}

		switch baseNeuron.Type {
		case "quantum":
			var qNeuron QuantumNeuron
			if err := json.Unmarshal(rawNeuron, &qNeuron); err != nil {
				return err
			}
			bp.QuantumNeurons[qNeuron.ID] = &qNeuron

		case "nca":
			var ncaNeuron Neuron
			if err := json.Unmarshal(rawNeuron, &ncaNeuron); err != nil {
				return err
			}
			bp.Neurons[ncaNeuron.ID] = &ncaNeuron

		case "cnn":
			var cnnNeuron Neuron
			if err := json.Unmarshal(rawNeuron, &cnnNeuron); err != nil {
				return err
			}
			// Ensure kernels are initialized; if not provided, initialize with default kernels
			if len(cnnNeuron.Kernels) == 0 {
				// Example: Initialize with default kernels
				cnnNeuron.Kernels = [][]float64{
					{0.2, 0.5}, // Default Kernel 0
					{0.3, 0.4}, // Default Kernel 1
				}
				if bp.Debug {
					fmt.Printf("CNN Neuron %d: No kernels provided. Initialized with default kernels.\n", cnnNeuron.ID)
				}
			}
			// Ensure activation is set to "relu"; if not, default to "relu"
			if cnnNeuron.Activation == "" {
				cnnNeuron.Activation = "relu"
				if bp.Debug {
					fmt.Printf("CNN Neuron %d: Activation not provided. Set to 'relu'.\n", cnnNeuron.ID)
				}
			}
			bp.Neurons[cnnNeuron.ID] = &cnnNeuron

		case "batch_norm":
			var bnNeuron Neuron
			if err := json.Unmarshal(rawNeuron, &bnNeuron); err != nil {
				return err
			}
			// Initialize BatchNormParams
			bnNeuron.BatchNormParams = &BatchNormParams{
				Gamma: 1.0,
				Beta:  0.0,
				Mean:  0.0,
				Var:   1.0,
			}
			// Ensure activation is set; default to "linear" if not provided
			if bnNeuron.Activation == "" {
				bnNeuron.Activation = "linear"
				if bp.Debug {
					fmt.Printf("BatchNorm Neuron %d: Activation not provided. Set to 'linear'.\n", bnNeuron.ID)
				}
			}
			bp.Neurons[bnNeuron.ID] = &bnNeuron

		default:
			var neuron Neuron
			if err := json.Unmarshal(rawNeuron, &neuron); err != nil {
				return err
			}
			// Initialize gate weights for LSTM neurons
			if neuron.Type == "lstm" {
				neuron.GateWeights = map[string][]float64{
					"input":  bp.RandomWeights(len(neuron.Connections)),
					"forget": bp.RandomWeights(len(neuron.Connections)),
					"output": bp.RandomWeights(len(neuron.Connections)),
					"cell":   bp.RandomWeights(len(neuron.Connections)),
				}
			}
			// Ensure activation is set; default to "linear" if not provided
			if neuron.Activation == "" {
				neuron.Activation = "linear"
				if bp.Debug {
					fmt.Printf("Neuron %d: Activation not provided. Set to 'linear'.\n", neuron.ID)
				}
			}
			bp.Neurons[neuron.ID] = &neuron
		}
	}

	return nil
}

// SaveToJSON saves the current Blueprint to a specified JSON file.
func (bp *Blueprint) SaveToJSON(fileName string) error {
	// Serialize the Blueprint to JSON
	data, err := json.MarshalIndent(bp, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to serialize Blueprint to JSON: %v", err)
	}

	// Write the JSON data to the specified file
	err = os.WriteFile(fileName, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write JSON to file '%s': %v", fileName, err)
	}

	fmt.Printf("Blueprint saved successfully to '%s'\n", fileName)
	return nil
}

// DownloadFile downloads a file from a URL and saves it locally.
func (bp *Blueprint) DownloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	// Check if the status is 200 OK
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: %s, status code: %d", url, resp.StatusCode)
	}

	// Create the output file
	out, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filepath, err)
	}
	defer out.Close()

	// Write response content to file
	_, err = io.Copy(out, resp.Body)
	return err
}

// UnzipFile unzips a .gz file into the specified target directory.
func (bp *Blueprint) UnzipFile(gzFile string, targetDir string) error {
	// Open the .gz file
	in, err := os.Open(gzFile)
	if err != nil {
		return fmt.Errorf("failed to open gzip file %s: %w", gzFile, err)
	}
	defer in.Close()

	// Create a gzip reader
	gz, err := gzip.NewReader(in)
	if err != nil {
		return fmt.Errorf("error creating gzip reader for %s: %v", gzFile, err)
	}
	defer gz.Close()

	// Determine output filename by removing .gz extension and adding to targetDir
	outFile := filepath.Join(targetDir, filepath.Base(gzFile[:len(gzFile)-3]))
	out, err := os.Create(outFile)
	if err != nil {
		return fmt.Errorf("failed to create output file %s: %w", outFile, err)
	}
	defer out.Close()

	// Copy the decompressed data to the output file
	_, err = io.Copy(out, gz)
	if err != nil {
		return fmt.Errorf("error during unzipping %s: %v", gzFile, err)
	}

	log.Printf("Unzipped %s successfully to %s\n", gzFile, outFile)
	return nil
}

// ToJSON serializes the Blueprint to a JSON string.
func (bp *Blueprint) SerializeToJSON() (string, error) {
	data, err := json.Marshal(bp)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// FromJSON deserializes the Blueprint from a JSON string.
func (bp *Blueprint) DeserializesFromJSON(data string) error {
	return json.Unmarshal([]byte(data), bp)
}

// getAllNeuronIDs retrieves the IDs of all neurons in the blueprint.
func (bp *Blueprint) getAllNeuronIDs() []int {
	neuronIDs := []int{}
	for id := range bp.Neurons {
		neuronIDs = append(neuronIDs, id)
	}
	return neuronIDs
}

// getRandomConnectionPair selects a random valid source and target neuron IDs for adding a connection.
// Returns -1, -1 if no valid pair is found.
func (bp *Blueprint) getRandomConnectionPair() (int, int) {
	neuronIDs := bp.getAllNeuronIDs()
	if len(neuronIDs) < 2 {
		return -1, -1
	}

	// Shuffle neuron IDs to randomize selection
	rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })

	for _, source := range neuronIDs {
		for _, target := range neuronIDs {
			if source == target {
				continue
			}
			// Check if connection already exists
			if bp.connectionExists(source, target) {
				continue
			}
			// We have a candidate
			return source, target
		}
	}
	return -1, -1
}

// getMaxFloat returns the maximum floating-point value.
func getMaxFloat() float64 {
	return math.MaxFloat64
}

func (bp *Blueprint) ValidateConnections() bool {
	visited := map[int]bool{}
	var dfs func(int)
	dfs = func(id int) {
		if visited[id] {
			return
		}
		visited[id] = true
		for _, conn := range bp.Neurons[id].Connections {
			dfs(int(conn[0]))
		}
	}
	for _, inputID := range bp.InputNodes {
		dfs(inputID)
	}
	for _, outputID := range bp.OutputNodes {
		if !visited[outputID] {
			fmt.Printf("Output Neuron %d is not connected.\n", outputID)
			return false
		}
	}
	return true
}
