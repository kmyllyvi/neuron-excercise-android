package com.example.simpleneuron

import kotlin.math.exp
import kotlin.random.Random // For more conventional random initialization

class SigmoidNeuron(
    // Initialize weights with small random values to break symmetry
    var w1: Double = Random.nextDouble(-0.5, 5.5),
    var w2: Double = Random.nextDouble(-0.5, 5.5),
    var bias: Double = Random.nextDouble(-0.5, 5.5),
    val learningRate: Double = 0.5 // faster rate = faster solution but less accuracy
) {

    private fun sigmoid(z: Double): Double {
        return 1.0 / (1.0 + exp(-z))
    }

    fun predict(x1: Double, x2: Double): Double {
        val weightedSum = w1 * x1 + w2 * x2 + bias
        return sigmoid(weightedSum)
    }

    // Target is 0.0 or 1.0
    // Returns the squared error for this specific instance
    fun learn(x1: Double, x2: Double, target: Double): Double {
        val prediction = predict(x1, x2)
        val error = target - prediction

        // Gradient of the loss w.r.t. the neuron's output before activation (z)
        // This includes the derivative of the sigmoid function: prediction * (1 - prediction)
        val delta = error * prediction * (1.0 - prediction)

        // Update weights and bias
        w1 += learningRate * delta * x1
        w2 += learningRate * delta * x2
        bias += learningRate * delta * 1.0 // Bias can be seen as a weight for an input that is always 1

        // For tracking, return the squared error
        return error * error
    }
}

// We'll adapt this logic into our Composable UI
fun main() {
    val sigmoidNeuron = SigmoidNeuron()

    // Example: Learn the OR function (target is 0.0 or 1.0)
    // (0,0) -> 0.0
    // (0,1) -> 1.0
    // (1,0) -> 1.0
    // (1,1) -> 1.0
    val trainingData = listOf( // Expanded to 10 samples
        Triple(0.0, 0.0, 0.0), Triple(0.0, 1.0, 1.0), Triple(1.0, 0.0, 1.0), Triple(1.0, 1.0, 1.0),
        Triple(0.0, 0.0, 0.0), Triple(0.0, 1.0, 1.0), Triple(1.0, 0.0, 1.0), Triple(1.0, 1.0, 1.0),
        Triple(0.0, 0.0, 0.0), Triple(0.0, 1.0, 1.0)
    )

    val epochs = 1000 // Sigmoid might need more epochs or careful learning rate/initialization

    println("Initial state: w1=${sigmoidNeuron.w1}, w2=${sigmoidNeuron.w2}, bias=${sigmoidNeuron.bias}")

    repeat(epochs) { epoch ->
        var currentEpochLoss = 0.0
        trainingData.shuffled().forEach { (x1, x2, target) ->
            currentEpochLoss += sigmoidNeuron.learn(x1, x2, target)
        }
        if ((epoch + 1) % 100 == 0) { // Print loss every 100 epochs
            println("Epoch ${epoch + 1}, Average Epoch Loss: ${currentEpochLoss / trainingData.size}")
            println("Neuron state: w1=${sigmoidNeuron.w1}, w2=${sigmoidNeuron.w2}, bias=${sigmoidNeuron.bias}")
        }
        // A simple convergence check (optional)
        if (currentEpochLoss / trainingData.size < 0.001 && epoch > 0) {
            println("Converged reasonably well at epoch ${epoch + 1}!")
            // break // Optional: stop if converged
        }
    }

    println("--- Testing after training ---")
    println("Final state: w1=${sigmoidNeuron.w1}, w2=${sigmoidNeuron.w2}, bias=${sigmoidNeuron.bias}")
    trainingData.forEach { (x1,x2,target) ->
        val prediction = sigmoidNeuron.predict(x1,x2)
        println("Input: ($x1,$x2), Target: $target, Prediction: ${"%.4f".format(prediction)} (Raw: $prediction)")
    }
}
