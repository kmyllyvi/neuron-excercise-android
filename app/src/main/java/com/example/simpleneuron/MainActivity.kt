package com.example.simpleneuron

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.simpleneuron.ui.theme.SimpleNeuronTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.isActive
import kotlin.random.Random

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SimpleNeuronTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    NeuronLearningScreen()
                }
            }
        }
    }
}

val inputPairs = List(10) { i ->
    val x1 = i * 0.1 // x1 will be 0.0, 0.1, 0.2, ..., 0.9
    val x2 = 0.5    // x2 is constant, for simplicity
    Pair(x1, x2)
}

val decliningTargetTrainingData = inputPairs.mapIndexed { index, pair ->
    val targetValue = 1.0 - (index.toDouble() / (inputPairs.size - 1).toDouble())
    Triple(pair.first, pair.second, targetValue)
}

val TARGET_AVG_EPOCH_LOSS = 0.005f
val MAX_AUTO_EPOCHS = 500

fun randomWeight(): Double = Random.nextDouble(-2.5, 2.5) // Kept wider range

// Helper to create initial random prediction points
fun createInitialRandomPredictions(numPoints: Int): List<Offset> {
    return List(numPoints) { i ->
        Offset(i.toFloat(), Random.nextDouble(0.0, 1.0).toFloat())
    }
}

@Composable
fun NeuronLearningScreen() {
    var w1 by remember { mutableStateOf(randomWeight()) }
    var w2 by remember { mutableStateOf(randomWeight()) }
    var bias by remember { mutableStateOf(randomWeight()) }

    val epochLossHistory = remember { mutableStateListOf<Float>() }
    var currentEpoch by remember { mutableIntStateOf(0) }
    // Initialize with random y-values for the visual effect
    val trialPredictionPoints = remember { mutableStateListOf<Offset>().apply {
        addAll(createInitialRandomPredictions(decliningTargetTrainingData.size))
    }}

    val scope = rememberCoroutineScope()
    var isAutoTraining by remember { mutableStateOf(false) }

    val currentTrainingData = decliningTargetTrainingData

    suspend fun trainOneEpochWithDelay(isAutomated: Boolean) {
        val currentEpochPredictionData = mutableListOf<Offset>()
        val tempNeuron = SigmoidNeuron(w1 = w1, w2 = w2, bias = bias)
        var epochSquaredErrorSum = 0.0

        for (i in currentTrainingData.indices) {
            val (x1Val, x2Val, targetVal) = currentTrainingData[i]
            if (isAutomated && !scope.isActive) break

            val prediction = tempNeuron.predict(x1Val, x2Val)
            val squaredErrorForTrial = tempNeuron.learn(x1Val, x2Val, targetVal)
            epochSquaredErrorSum += squaredErrorForTrial

            w1 = tempNeuron.w1
            w2 = tempNeuron.w2
            bias = tempNeuron.bias

            currentEpochPredictionData.add(Offset(i.toFloat(), prediction.toFloat()))

            val delayTime = if (isAutomated) 10L else 150L
            delay(delayTime)
        }

        if (scope.isActive) {
            val averageEpochLoss = (epochSquaredErrorSum / currentTrainingData.size).toFloat()
            epochLossHistory.add(averageEpochLoss)
            // CORRECTED THE TYPO HERE:
            if (epochLossHistory.size > 20) epochLossHistory.removeAt(0)
            currentEpoch++

            trialPredictionPoints.clear() // Clear the initial random (or previous epoch's) points
            trialPredictionPoints.addAll(currentEpochPredictionData) // Add actual neuron predictions
        }
    }

    LaunchedEffect(isAutoTraining) {
        if (isAutoTraining) {
            while (isActive &&
                currentEpoch < MAX_AUTO_EPOCHS &&
                (epochLossHistory.lastOrNull() ?: Float.MAX_VALUE) > TARGET_AVG_EPOCH_LOSS) {
                trainOneEpochWithDelay(isAutomated = true)
            }
            isAutoTraining = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Neuron Learning: Declining Target", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))

        Row(horizontalArrangement = Arrangement.SpaceEvenly, modifier = Modifier.fillMaxWidth()) {
            Button(onClick = {
                isAutoTraining = !isAutoTraining
            }) {
                Text(if (isAutoTraining) "Stop Training" else "Start Training")
            }
            Button(onClick = {
                w1 = randomWeight()
                w2 = randomWeight()
                bias = randomWeight()
                currentEpoch = 0
                epochLossHistory.clear()
                // Reset to initial random points for the visual effect
                trialPredictionPoints.clear()
                trialPredictionPoints.addAll(createInitialRandomPredictions(currentTrainingData.size))
                isAutoTraining = false
            }, enabled = !isAutoTraining) {
                Text("Reset")
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Text("Epoch: $currentEpoch")
        Text("Last Avg Epoch Loss: ${if (epochLossHistory.isNotEmpty()) "%.5f".format(epochLossHistory.last()) else "N/A"}")

        Spacer(modifier = Modifier.height(16.dp))

        Text("Predictions vs. Declining Target:", style = MaterialTheme.typography.titleMedium)
        Canvas(modifier = Modifier.fillMaxWidth().height(250.dp).padding(8.dp)) {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val padding = 20.dp.toPx()
            val plotWidth = canvasWidth - 2 * padding
            val plotHeight = canvasHeight - 2 * padding

            drawRect(Color.LightGray, style = Stroke(1.dp.toPx()))
            drawLine(Color.Gray, Offset(padding, padding + plotHeight), Offset(padding + plotWidth, padding + plotHeight))
            drawLine(Color.Gray, Offset(padding, padding), Offset(padding, padding + plotHeight))

            val maxTrials = (currentTrainingData.size - 1).toFloat().coerceAtLeast(1f)
            val maxOutputValue = 1.0f

            trialPredictionPoints.forEach { point ->
                val plotX = padding + (point.x / maxTrials) * plotWidth
                // Y value is already between 0.0 and 1.0 from createInitialRandomPredictions or neuron output
                val plotY = padding + plotHeight - (point.y.coerceIn(0f, 1f) / maxOutputValue) * plotHeight
                drawCircle(
                    color = Color.Magenta,
                    radius = 5.dp.toPx(),
                    center = Offset(plotX.coerceIn(padding, padding + plotWidth), plotY.coerceIn(padding, padding + plotHeight))
                )
            }

            if (currentTrainingData.size > 1) {
                for (i in 0 until currentTrainingData.size - 1) {
                    val target1 = currentTrainingData[i].third.toFloat()
                    val target2 = currentTrainingData[i+1].third.toFloat()
                    val x1 = padding + (i.toFloat() / maxTrials) * plotWidth
                    val y1 = padding + plotHeight - (target1 / maxOutputValue) * plotHeight
                    val x2 = padding + ((i + 1).toFloat() / maxTrials) * plotWidth
                    val y2 = padding + plotHeight - (target2 / maxOutputValue) * plotHeight
                    drawLine(
                        color = Color.Cyan,
                        start = Offset(x1.coerceIn(padding, padding + plotWidth), y1.coerceIn(padding, padding + plotHeight)),
                        end = Offset(x2.coerceIn(padding, padding + plotWidth), y2.coerceIn(padding, padding + plotHeight)),
                        strokeWidth = 2.dp.toPx()
                    )
                }
            }
            currentTrainingData.forEachIndexed { index, triple ->
                val targetValue = triple.third.toFloat()
                val plotXTarget = padding + (index.toFloat() / maxTrials) * plotWidth
                val plotYTarget = padding + plotHeight - (targetValue / maxOutputValue) * plotHeight
                drawCircle(
                    color = Color.Cyan,
                    radius = 2.dp.toPx(),
                    center = Offset(plotXTarget.coerceIn(padding, padding + plotWidth), plotYTarget.coerceIn(padding, padding + plotHeight))
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    SimpleNeuronTheme {
        NeuronLearningScreen()
    }
}

