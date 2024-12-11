package blueprint

import (
	"fmt"

	"golang.org/x/exp/rand"
)

// blueprint.go

// HillClimbWeightUpdate performs random perturbations on the network's weights.
// It keeps the changes if the performance improves.
func (bp *Blueprint) HillClimbWeightUpdate(sessions []Session) bool {
	// Clone the current blueprint to test changes
	candidateBP := bp.Clone()
	if candidateBP == nil {
		fmt.Println("Failed to clone blueprint for weight update.")
		return false
	}

	// Define the maximum change per weight
	const maxWeightChange = 0.1

	// Randomly select a neuron and a connection to perturb
	neuronIDs := bp.getAllNeuronIDs()
	if len(neuronIDs) == 0 {
		fmt.Println("No neurons available for weight update.")
		return false
	}

	// Select a random neuron (excluding input neurons)
	var targetNeuron *Neuron
	for {
		randomNeuronID := neuronIDs[rand.Intn(len(neuronIDs))]
		if !bp.isInputNode(randomNeuronID) {
			targetNeuron = candidateBP.Neurons[randomNeuronID]
			if targetNeuron != nil && len(targetNeuron.Connections) > 0 {
				break
			}
		}
	}

	// Select a random connection from the target neuron
	connIndex := rand.Intn(len(targetNeuron.Connections))
	originalWeight := targetNeuron.Connections[connIndex][1]

	// Perturb the weight by a small random value
	perturbation := (rand.Float64()*2 - 1) * maxWeightChange // Random change between -maxWeightChange and +maxWeightChange
	targetNeuron.Connections[connIndex][1] += perturbation

	// Evaluate the candidate blueprint's performance
	exactAcc, generousAcc, forgivenessAcc, _, _, _ := candidateBP.EvaluateModelPerformance(sessions)

	// Evaluate the current blueprint's performance
	currentExactAcc, currentGenerousAcc, currentForgivenessAcc, _, _, _ := bp.EvaluateModelPerformance(sessions)

	// Determine if the candidate is better
	improved := false
	if exactAcc > currentExactAcc || generousAcc > currentGenerousAcc || forgivenessAcc > currentForgivenessAcc {
		improved = true
	}

	if improved {
		// Accept the changes
		*bp = *candidateBP
		if bp.Debug {
			fmt.Printf("Weight Update Accepted: Neuron %d Connection %d Weight changed from %.4f to %.4f\n",
				targetNeuron.ID, connIndex, originalWeight, targetNeuron.Connections[connIndex][1])
		}
		return true
	} else {
		// Reject the changes
		if bp.Debug {
			fmt.Printf("Weight Update Rejected: Neuron %d Connection %d Weight remains at %.4f\n",
				targetNeuron.ID, connIndex, originalWeight)
		}
		return false
	}
}
