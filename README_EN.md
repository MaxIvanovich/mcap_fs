# mcap_fs #
**mcap_fs - Market Charts Analisis and Prediction (first study)**

## Task ##
***Create neural network for prediction of the price direction for next day, based on previous day(s)***

## First study ##

### Neural network structure ###

* **Six input neurons for the folowing parameters:**
	* open - day open
	* high - day high
	* low - day low
	* close - day close
	* volume - day volume
	* direction - growth/decline
* **Six hidden neurons**
* **Two output neurons:**
	* Growth neuron, active when growth
	* decline neuron, active when decline

#### Activation function ####
Sigmoid for all neurons
