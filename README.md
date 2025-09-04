### the MOST SIMPLE AND BASIC Neuron which Learns and solves one little ML Problem (a linear regression)
created with Android Studio and help of Gemini 2.5

####UI Behaviour
Press "start learning" and see how the dots start arringing towards the ideal target line.
You can see the epoc counter representing how many adaptation cycles the app needs until suffice "avg epoc loss" rate has been achieved. This is a configurabele value. When the value is reached the app stops learning. For a new round hit "reset"

####The "engine" (the SigmoidNeuron's learning principle) works like this:
1. Predict:<br>
  It takes two input values, combines them with its internal "weights" and "bias" (its current knowledge), and passes this through a sigmoid function to produce a prediction between 0 and 1.
2. Compare & Calculate Error:<br>
  It compares this prediction to the "target" (correct) value for those inputs. The difference is the error.
3. Learn (Adjust Weights):<br>
   It uses this error to slightly adjust its weights and bias. If it predicted too high, it nudges them to predict lower next time for similar inputs (and vice-versa). This adjustment is guided by       the learning rate and the derivative of the sigmoid function (which tells it how sensitive its output is to changes in its internal sum).Essentially, it's: predict, see how wrong it was, and tweak    its internal settings to be less wrong next time.
