package blueprint

import (
	"fmt"
	"math"
	"math/rand"
)

// Neuron represents a single neuron in the network
type Neuron struct {
	ID          int         `json:"id"`
	Type        string      `json:"type"`         // Dense, RNN, LSTM, CNN, etc.
	Value       float64     `json:"value"`        // Current value
	Bias        float64     `json:"bias"`         // Default: 0.0
	Connections [][]float64 `json:"connections"`  // [source_id, weight]
	Activation  string      `json:"activation"`   // Activation function
	LoopCount   int         `json:"loop_count"`   // For RNN/LSTM loops
	WindowSize  int         `json:"window_size"`  // For CNN
	DropoutRate float64     `json:"dropout_rate"` // For Dropout
	BatchNorm   bool        `json:"batch_norm"`   // Apply batch normalization
	Attention   bool        `json:"attention"`    // Apply attention mechanism
	Kernels     [][]float64 `json:"kernels"`      // Multiple kernels for CNN neurons
	// Additional fields for LSTM
	CellState   float64              // For LSTM cell state
	GateWeights map[string][]float64 // Weights for LSTM gates

	NeighborhoodIDs []int  `json:"neighborhood"` // IDs of neighboring neurons (for NCA)
	UpdateRules     string `json:"update_rules"` // Rules for updating (e.g., Sum, Average)
}

// ProcessNeuron processes a single neuron based on its type
func (bp *Blueprint) ProcessNeuron(neuron *Neuron, inputs []float64, timestep int) {
	// Skip processing input neurons
	if neuron.Type == "input" {
		return
	}

	switch neuron.Type {
	case "nca":
		bp.ProcessNCANeuron(neuron)
	case "rnn":
		bp.ProcessRNNNeuron(neuron, inputs)
	case "lstm":
		bp.ProcessLSTMNeuron(neuron, inputs)
	case "cnn":
		bp.ProcessCNNNeuron(neuron, inputs)
	case "dropout":
		bp.ApplyDropout(neuron)
	case "batch_norm":
		bp.ApplyBatchNormalization(neuron, 0.0, 1.0) // Example mean/variance
	case "attention":
		// Handled separately in Forward method
		fmt.Printf("Attention Neuron %d processed\n", neuron.ID)
	default:
		// Default dense neuron behavior
		bp.ProcessDenseNeuron(neuron, inputs)
	}
}

// ProcessDenseNeuron handles standard dense neuron computation
func (bp *Blueprint) ProcessDenseNeuron(neuron *Neuron, inputs []float64) {
	sum := neuron.Bias
	for _, input := range inputs {
		sum += input
	}
	neuron.Value = bp.ApplyScalarActivation(sum, neuron.Activation)
	fmt.Printf("Dense Neuron %d: Value=%f\n", neuron.ID, neuron.Value)
}

// ProcessRNNNeuron updates an RNN neuron over multiple time steps
func (bp *Blueprint) ProcessRNNNeuron(neuron *Neuron, inputs []float64) {
	// Simple RNN implementation with separate weight for previous value
	sum := neuron.Bias
	for _, input := range inputs {
		sum += input // Already includes weights from connections
	}
	// Add weighted previous value (assuming weight of 1.0 for simplicity)
	sum += neuron.Value * 1.0
	neuron.Value = bp.ApplyScalarActivation(sum, neuron.Activation)
	fmt.Printf("RNN Neuron %d: Value=%f\n", neuron.ID, neuron.Value)
}

// ProcessLSTMNeuron updates an LSTM neuron with gating
func (bp *Blueprint) ProcessLSTMNeuron(neuron *Neuron, inputs []float64) {
	// Standard LSTM cell implementation with weights
	var (
		inputGate  float64
		forgetGate float64
		outputGate float64
		cellInput  float64
	)

	weights := neuron.GateWeights
	inputSize := len(inputs)

	// Compute gates with weights
	for i := 0; i < inputSize; i++ {
		inputGate += inputs[i] * weights["input"][i]
		forgetGate += inputs[i] * weights["forget"][i]
		outputGate += inputs[i] * weights["output"][i]
		cellInput += inputs[i] * weights["cell"][i]
	}

	inputGate = Sigmoid(inputGate + neuron.Bias)
	forgetGate = Sigmoid(forgetGate + neuron.Bias)
	outputGate = Sigmoid(outputGate + neuron.Bias)
	cellInput = Tanh(cellInput + neuron.Bias)

	// Update cell state and output
	neuron.CellState = neuron.CellState*forgetGate + cellInput*inputGate
	neuron.Value = Tanh(neuron.CellState) * outputGate

	fmt.Printf("LSTM Neuron %d: Value=%f, CellState=%f\n", neuron.ID, neuron.Value, neuron.CellState)
}

// ProcessCNNNeuron applies convolutional behavior using the neuron's predefined kernels
func (bp *Blueprint) ProcessCNNNeuron(neuron *Neuron, inputs []float64) {
	if len(neuron.Kernels) == 0 {
		fmt.Printf("CNN Neuron %d: No kernels defined. Setting value to 0.\n", neuron.ID)
		neuron.Value = 0.0
		return
	}

	// Iterate over each kernel assigned to the neuron
	convolutionOutputs := []float64{}
	for k, kernel := range neuron.Kernels {
		kernelSize := len(kernel)
		if len(inputs) < kernelSize {
			fmt.Printf("CNN Neuron %d: Skipping kernel %d due to insufficient inputs (required: %d, got: %d)\n", neuron.ID, k, kernelSize, len(inputs))
			continue
		}

		// Perform convolution for the current kernel
		for i := 0; i <= len(inputs)-kernelSize; i++ {
			sum := neuron.Bias
			for j := 0; j < kernelSize; j++ {
				sum += inputs[i+j] * kernel[j]
			}
			activatedValue := bp.ApplyScalarActivation(sum, neuron.Activation)
			convolutionOutputs = append(convolutionOutputs, activatedValue)
			fmt.Printf("CNN Neuron %d: Kernel %d Output[%d]=%f\n", neuron.ID, k, i, activatedValue)
		}
	}

	// Handle cases where no valid convolution outputs were generated
	if len(convolutionOutputs) == 0 {
		fmt.Printf("CNN Neuron %d: No valid convolution outputs. Setting value to 0.\n", neuron.ID)
		neuron.Value = 0.0
		return
	}

	// Aggregate the convolution outputs (e.g., by taking the mean)
	aggregate := 0.0
	for _, v := range convolutionOutputs {
		aggregate += v
	}
	neuron.Value = aggregate / float64(len(convolutionOutputs))
	fmt.Printf("CNN Neuron %d: Aggregated Value=%f\n", neuron.ID, neuron.Value)
}

// ApplyDropout randomly zeroes out a neuron's value
func (bp *Blueprint) ApplyDropout(neuron *Neuron) {
	if rand.Float64() < neuron.DropoutRate {
		neuron.Value = 0
		fmt.Printf("Dropout Neuron %d: Value set to 0\n", neuron.ID)
	} else {
		fmt.Printf("Dropout Neuron %d: Value retained as %f\n", neuron.ID, neuron.Value)
	}
}

// ApplyBatchNormalization normalizes the neuron's value
func (bp *Blueprint) ApplyBatchNormalization(neuron *Neuron, mean, variance float64) {
	neuron.Value = (neuron.Value - mean) / math.Sqrt(variance+1e-7)
	fmt.Printf("BatchNorm Neuron %d: Normalized Value=%f\n", neuron.ID, neuron.Value)
}

// ApplyAttention adjusts neuron values based on attention weights
func (bp *Blueprint) ApplyAttention(neuron *Neuron, inputs []float64, attentionWeights []float64) {
	// Compute attention-weighted sum
	sum := neuron.Bias
	for i, input := range inputs {
		sum += input * attentionWeights[i]
	}
	neuron.Value = bp.ApplyScalarActivation(sum, neuron.Activation)
	fmt.Printf("Attention Neuron %d: Value=%f\n", neuron.ID, neuron.Value)
}

// ComputeAttentionWeights computes attention weights for the given inputs
func (bp *Blueprint) ComputeAttentionWeights(neuron *Neuron, inputs []float64) []float64 {

	// Simple scaled dot-product attention
	queries := inputs
	keys := inputs

	// Compute attention scores
	scores := make([]float64, len(inputs))
	for i := range inputs {
		scores[i] = queries[i] * keys[i] // Dot product
	}

	// Apply softmax to get weights
	attentionWeights := Softmax(scores)
	fmt.Printf("Attention Neuron %d: Weights=%v\n", neuron.ID, attentionWeights)
	return attentionWeights
}

// ApplySoftmax applies the Softmax function to all output neurons collectively
func (bp *Blueprint) ApplySoftmax() {
	outputValues := []float64{}
	for _, id := range bp.OutputNodes {
		if neuron, exists := bp.Neurons[id]; exists {
			outputValues = append(outputValues, neuron.Value)
		}
	}

	// Apply Softmax to the collected output values
	softmaxValues := Softmax(outputValues)

	// Assign the Softmaxed values back to the output neurons
	for i, id := range bp.OutputNodes {
		if neuron, exists := bp.Neurons[id]; exists {
			neuron.Value = softmaxValues[i]
			fmt.Printf("Softmax Applied to Neuron %d: Value=%f\n", id, neuron.Value)
		}
	}
}

func (bp *Blueprint) ProcessNCANeuron(neuron *Neuron) {
	// Gather values from neighboring neurons
	neighborValues := []float64{}
	for _, neighborID := range neuron.NeighborhoodIDs {
		if neighbor, exists := bp.Neurons[neighborID]; exists {
			neighborValues = append(neighborValues, neighbor.Value)
		}
	}

	// Apply update rules
	var newValue float64
	switch neuron.UpdateRules {
	case "sum":
		for _, value := range neighborValues {
			newValue += value
		}
	case "average":
		sum := 0.0
		for _, value := range neighborValues {
			sum += value
		}
		newValue = sum / float64(len(neighborValues))
	default:
		fmt.Printf("Unknown update rule for NCA Neuron %d\n", neuron.ID)
		return
	}

	// Apply activation function
	neuron.Value = bp.ApplyScalarActivation(newValue+neuron.Bias, neuron.Activation)
	fmt.Printf("NCA Neuron %d: Value=%f\n", neuron.ID, neuron.Value)
}

func (bp *Blueprint) InitializeKernel(kernelSize int) []float64 {
	kernel := make([]float64, kernelSize)
	for i := range kernel {
		kernel[i] = rand.Float64() // Initialize with random weights between 0 and 1
	}
	return kernel
}
