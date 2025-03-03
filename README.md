# Intelligent PID Tuning for Electrical Drives Using Neural Networks

## Project Overview
The repository integrates several key components:
- **PID Controller & Numerical Methods:** Implements PID control using numerical differentiation and integration.
- **Neural Network Tuning:** A feedforward neural network dynamically adjusts PID gains (Kp, Ki, Kd).
- **System Plant & Signal Generation:** Simulates a linear or non-linear plant and generates reference signals.
- **Simulation & Visualization:** Runs the simulation loop and plots results to assess performance.
- **AutoRegressive (AR) Denoising:** Uses AR models (in MATLAB and Python) to filter noise from input signals.
## Repository Structure and Key Components

The repository is organized into several key modules and folders, each addressing specific functionalities of the project. Below is an overview of the structure and the role of each component.

### 1. PID Controller and Neural Network Tuning

#### a. PID Controller Components
- **`numeric_dif.py`**  
  - **Class:** `Diff_num`  
  - **Functionality:** Computes numerical differentiation using the formula:  
    ```
    derivative = (new_val - prev_val) / sample_rate
    ```
  - **Note:** Updates the previous value for subsequent calculations.

- **`numeric_intg.py`**  
  - **Class:** `Intg_num`  
  - **Functionality:** Implements numerical integration using the trapezoidal rule:  
    ```
    integral = acc + (sample_rate / 2) * (prev_val + new_val)
    ```
  - **Note:** Accumulates the integral result and updates the previous value.

- **`pid_cntrl.py`**  
  - **Class:** `PID_CNTRL`  
  - **Functionality:**  
    - Computes the PID control output using:
      ```
      output = Kp * error + Ki * integral + Kd * derivative
      ```
    - **Methods:**
      - `_proc(new_e)`: Computes the PID output based on the current error.
      - `_update_k(Kp, Kd, Ki)`: Dynamically updates the PID coefficients.

#### b. Neural Network-PID Interface
- **`NN_PID_intrface.py`**  
  - **Class:** `NN_PID_interface`  
  - **Functionality:**  
    - Interfaces the PID controller with the neural network.
    - Computes gradients to adjust PID gains using:
      ```
      dJ_dyn = -2 * new_error * sys_coef
      dJ_dk  = dJ_dyn * prev_error
      ```
    - Updates stored error values for iterative tuning.

#### c. Neural Network Components
- **`neuron.py`**  
  - **Functionality:** Implements a feedforward neural network for tuning PID gains.
  - **Key Classes:**
    - **`neuron_layers`:**  
      - Manages a multilayer network.
      - Implements forward propagation (`forward` / `calc_output`) and backpropagation (`back_prop`).
    - **`neuron_layer`:**  
      - Represents a single layer with weights (`w`) and biases (`b`).
      - Applies activation functions (ReLU, Sigmoid) and performs backpropagation for weight updates.
    - **`output_layer`:**  
      - Specializes in computing the network’s output error using Mean Squared Error (MSE) loss.
  - **Input/Output:**  
    - Input: Concatenation of past errors, control signals, plant outputs, and reference signals.
    - Output: Updated PID gains `[Kp, Kd, Ki]`.

### 2. System Plant and Signal Generation

#### a. Plant Simulation
- **`plant.py`**  
  - **Class:** `Plant`  
  - **Functionality:** Simulates system dynamics.
    - **Linear System Equation:**
      ```
      y(n+1) = 0.998 * y(n) + 0.232 * u(n)
      ```
    - **Non-Linear System Equation:**
      ```
      y(n+1) = 0.9 * y(n) - 0.001 * y(n-1)^2 + u(n) + sin(u(n-1))
      ```
  - **Note:** Updates state variables for previous outputs and inputs.

#### b. Reference Signal Generation
- **`system_input.py`**  
  - **Function:** `_input_gen`  
  - **Functionality:** Generates a step-wise reference signal.
    - For linear systems: Increases by a fixed increment every 60 time steps.
    - For non-linear systems: Uses varying increments to simulate complex behavior.

### 3. Simulation and Data Visualization

#### a. Simulation Script
- **`Sim.py`**  
  - **Functionality:**  
    - Serves as the main simulation loop.
    - **Workflow:**
      1. Initializes the PID controller, neural network, and plant from `System_Params.py`.
      2. Iterates over time steps to:
         - Generate reference signals.
         - Compute control signals using the PID controller.
         - Update PID gains via the neural network (if enabled).
         - Process plant output and compute error.
         - Backpropagate error to adjust the neural network.
         - Update FIFO buffers for neural network inputs.
         - Store simulation data for visualization.
      3. Calls functions from `graph_sim.py` to plot the results.

#### b. Graphing Module
- **`graph_sim.py`**  
  - **Functionality:**  
    - Plots and saves graphs showing:
      - Reference vs. plant output.
      - Error over time.
      - Evolution of PID coefficients.
  - **Note:** Saves plots in directories based on whether the neural network is enabled (`wNN`) and the type of system (linear/non-linear).

### 4. System Parameters and Testing

#### a. System Parameters
- **`System_Params.py`**  
  - **Functionality:**  
    - Centralizes configuration and initialization.
    - **Defines:**
      - Initial PID coefficients (`Kp`, `Ki`, `Kd`).
      - Neural network architecture (hidden layers, learning rate).
      - Instantiation of the PID controller, plant, neural network (`Deep_NN`), and the NN-PID interface (`Nn_pid_intrf`).
      - Arrays to store simulation data (errors, control inputs, outputs, and PID coefficients).

#### b. Unit Testing
- **`unit_test.py`**  
  - **Functionality:**  
    - Tests the dimensions and concatenation of arrays used as inputs for the neural network.
    - Ensures data formatting is correct before running the full simulation.

### 5. AutoRegressive (AR) and RNN Models for Denoising

#### a. MATLAB AR Models
- **`Ar_Model.m`**  
  - **Functionality:**  
    - Uses an AR model (order 8) to predict and denoise a noisy sine wave.
    - Plots comparisons between:
      - The original signal.
      - The noisy signal.
      - The AR model’s output.
- **`data_prep.m`**  
  - **Functionality:**  
    - Generates a sine wave and adds Gaussian noise to simulate sensor data.

#### b. Python AR Model
- **`ar_model.py`**  
  - **Functionality:**  
    - Utilizes the Statsmodels library to fit an AutoRegressive model (lags=5).
    - Splits the noisy sine wave data into training and testing sets.
    - Predicts and plots the denoised signal against the actual noisy data.

## Required packages

Installation of numpy (Matrix calculation libary):

    $ pip install numpy

Installation of Matplotlib (Visiual Graphing libary):

    $ pip install matplotlib

Installation of Logging (Easy logging libary):

    $ pip install logging

To run the simulation run the python file named (`\Sim.py`)
To try out different parameters check (`\System_Params`)
To check out derivation of calculations used in this code check (`\PID_Tuning_wDNN.pdf`) or referenced resarch paper.

#### Signals
(/Sim.py)
Simulation signals can be tracked in the file (`Sim.py`):
* `ref[t]`: Reference signal  @ sim_time: t
* `u_signal[t]`: PID Output Signal @ sim_time: t
* `k_coefs[t]`: Pid coefficents @ sim_time: t
* `error_t_1_signal[t]`: Reference - Plant output  @ sim_time: t
* `out_t_1_signal[t]`: Plant output @ sim_time: t

#### Constructor parameters:
(/System_Params.py)
Simulation parameters are modifable to some degree in the file `System_Params.py`:

* `is_linear`: Paper describes two different systems. Non-linear or Linear. Make this parameter True for linear choice and false for otherwise. (type : bool)
* `w_NN`: Enable neural network to update coefficents. w_NN = True to enable NN, false for bypassing the NN. (type: bool)
* `lr` : learning rate of the neural network. (type : float)
* `hidden_layers`: Number of nodes in each layer (type : list)
Restriction being:
It should be an list, 1st index should be 12, last index should be 3.hidden_layers[0] = 12 hidden_layers[-1] = 3
* `Kp`: Inital propotional coefficent of the PID. (type : float)
* `Ki`: Initial integralcoefficent of the PID (type : float)
* `Kd`: Initial differential coefficent of the PID (type : float)

#### Objects:
(/PID)
* `numeric_dif()` :Containts attributes of `prev_val` calculates the difference with **Limit derivation of derivative**. Method of this object is `diff()` which calculates the differential result of the given values regarding the previous value.
* `numeric_intg()` : Containts attributes of `acc` and `prev_val` calculates the integral with **Trapozoidal Method** . Method of this object is `intg()` which calculates the differential result of the given values regarding the previous value.
* `PID_CNTRL()` : Containts `numeric_intg` and  `numeric_dif()` objects as attributes. Two method exist for this object `_proc()` and `update_k()`. First method calculates output of the PID by described methods of numeric methods of differential and integrations. 
(/Neuron)
* `Neuron_layer()`: Single neuron layer. Contains the method of `forward()` and `backprop()`: First method mentioned is foward propagation, and secondone is backpropagation. Needed variables to calculate partial differentials are stored in class variables such as attributes `self.w`, `self.x`, `self.b`.
* `Neuron_Layers()`: Contains `Neuron_layer()` objects. Methods of neuron_layer are `calc_output()` and `backprop()`. Backprop calculates partial derivative of each weigths and biases with reverse direction by calling `Neuron_layer().backprop()` of single layers backproapgation method one by one. Calculate output method is simallry do this in forward direction to return output of the neuron depedning on the output.
