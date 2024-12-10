package main

import (
        "fmt"
        "math/rand"
)

type Perceptron struct {
        weights []float64
        bias    float64
        learningRate float64
}

func NewPerceptron(inputSize int, learningRate float64) *Perceptron {
        weights := make([]float64, inputSize)
        for i := range weights {
                weights[i] = rand.Float64()
        }
        bias := rand.Float64()
        return &Perceptron{weights, bias, learningRate}
}

func (p *Perceptron) Activate(inputs []float64) float64 {
        sum := 0.0
        for i, weight := range p.weights {
                sum += inputs[i] * weight
        }
        sum += p.bias
        return max(0, sum)
}

func (p *Perceptron) Train(inputs []float64, target float64) {
        output := p.Activate(inputs)
        error := target - output
        for i, input := range inputs {
                p.weights[i] += p.learningRate * error * input
        }
        p.bias += p.learningRate * error
}

func main() {
        p := NewPerceptron(2, 0.1)

        trainingData := [][]float64{
                {0, 0}, {0, 1}, {1, 0}, {1, 1},
        }
        targets := []float64{0, 1, 1, 1}

        for epoch := 0; epoch < 10000; epoch++ {
                for i, input := range trainingData {
                        p.Train(input, targets[i])
                }
        }

        for _, input := range trainingData {
                output := p.Activate(input)
                fmt.Printf("Input: %v, Output: %.2f\n", input, output)
        }
}

func max(a, b float64) float64 {
        if a > b {
                return a
        }
        return b
}
