# mcap_fs #
**mcap_fs - Market Charts Analisis and Prediction (first study)**

## Task ##
***Create neural network for prediction of the price direction for next day, based on previous day(s)***

## First study ##

### Neural network structure ###

* **Six input neurons for the following parameters:**
	* open - day open
	* high - day high
	* low - day low
	* close - day close
	* volume - day volume
	* direction - growth/decline
* **Six hidden neurons**
* **One output neuron:**
	* values close to 1 means growth
	* values close to -1 means decline

#### Activation function ####
Hyperbolic tangent for output neuron, sigmoid for all others