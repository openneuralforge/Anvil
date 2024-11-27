package blueprint

import (
	"math"
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
