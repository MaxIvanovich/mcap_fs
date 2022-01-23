# mcap_fs #
**mcap_fs - Market Charts Analisis and Prediction (first study)**

## Task ##
***Create neural network for prediction of the price direction for next day, based on previous day(s)***

---

## First study ##

### Neural network structure ###

* **Inputs parameters for days:**
	* open - day open
	* high - day high
	* low - day low
	* close - day close
	* volume - day volume
	* direction - growth/decline
* **On first step hidden neurons quantity is equal to input parameters**
* **One output neuron:**
	* values close to 1 means growth
	* values close to -1 means decline

#### Activation function ####
Hyperbolic tangent for output neuron, sigmoid for all others

### First study results ###
Testing shown about 55% of the network efficiency

---

## Second study ##
Soon...