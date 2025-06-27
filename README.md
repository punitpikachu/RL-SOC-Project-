# RL SOC Project

## Week 1 - Python, PyGame, OOP, Snake Game Assignment 

### Python
Learned basic Python syntax from [here](https://www.w3schools.com/python/). 

### PyGame 
Pygame is a set of Python modules designed for writing 2D games. This allows you to create fully featured games and multimedia programs in the python language. Pygame is highly portable and runs on nearly every platform and operating system.
[PyGame Tutorial](https://www.geeksforgeeks.org/python/pygame-tutorial/)

**Topics Covered**
- Creating, Naming, Resizing PyGame window and changing PyGame icon
- How to set up Game Loop in PyGame, Creating Surfaces, Time functions 
-  Event Handling, Input Handling - getting keyboard and mouse inputs from the player 
- Displaying Text on screen, accepting text inputs
- displaying images, rotating, scaling and moving an image, using mouse to scale and rotate images 
- Advanced PyGame - creating buttons in a game, moving an object, adding boundary to an object, collision detection, creating and controlling sprites, color breezing effects and playing audio files 

### Object Oriented Programming (OOP)

- Object-Oriented Programming (OOP) is a programming paradigm that organizes code around objects, which are instances of classes—blueprints that define the data (attributes) and behavior (methods) of those objects.
- Understanding the difference between Procedural-Oriented-Approach and Object-Oriented-Approach

**Core Concpets of OOP**

1. Class
A class is a blueprint or template for creating objects. It defines the attributes (data) and behaviors (methods) an object will have.

2. Object
An object is an instance of a class. You can create multiple objects from the same class, each with its own data.

3. Encapsulation
Encapsulation means bundling data and methods that operate on that data within one unit (class). It also often implies hiding internal details (data hiding) and exposing only what’s necessary via interfaces.

4. Inheritance
Inheritance allows a class (child/subclass) to inherit attributes and methods from another class (parent/superclass). It promotes code reuse.

5. Polymorphism
Polymorphism allows different classes to implement the same method differently. This supports dynamic behavior and makes code more flexible.

6. Abstraction
Abstraction hides complex implementation details and shows only the necessary features to the user.

You achieve this by designing interfaces or base classes, and letting the user focus on "what it does" instead of "how it works."

## Snake Game Assignment 

Using the model of OOP we had to implement Pygame and create the snake game. In this game, the main objective of the player is to catch the maximum number of fruits without hitting the wall or 
itself.

- Extra Features - Preset and Custom Difficulty - player can set the speed of the snake

[Snake Game assignment submission](https://docs.google.com/document/d/1T-Tr6LTkTH4Og5xfL418xm7WYrJTzfpMjsn0bQtEZvQ/edit?usp=sharing)

## Week 2 - Deep Learning, Neural Networks, CNNs, Pytorch, MNIST Digit Classifier using Pytorch 

### Neural Networks, CNNs

- Got introduced to the idea behind neural networks and their mechanism. Basics of cost function, weights, biases, activations, sigmoid function and ReLU through 3b1b series on neural networks - [neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) and Deep Learning specialisation course by Andrew Ng -[Deep Learning Specialisation](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
- Referred to CS231n course taken by Andrej Karpathy for few topics - Backpropogation, CNNs, Localization and Detection
- A convolution is an operation where a small filter slides over input data, multiplying and summing values to extract local patterns. It focuses on nearby relationships rather than treating all inputs equally. A Convolutional Neural Network (CNN) stacks multiple such convolution layers, allowing the model to learn hierarchical patterns efficiently while keeping the number of parameters low. CNNs exploit locality and parameter sharing to process structured data more effectively.
- [What is Convolution](https://youtu.be/KuXjwB4LzSA), [CNNs for computer vision](https://www.youtube.com/watch?v=oGpzWAlP5p0)

### Pytorch

PyTorch is a deep learning framework that combines tensor computation with automatic differentiation, enabling flexible, GPU-accelerated model building and training using dynamic computation graphs.

- Learned Pytorch with [this](https://youtu.be/OIenNRt2bjg) tutorial

### MNIST Digit Classifier Using Pytorch

- In this project, we built an MNIST digit classifier using PyTorch to recognize handwritten digits (0–9). The model was trained on the MNIST dataset using both a simple fully connected neural network and a convolutional neural network (CNN) to compare performance

- The CNN achieved higher accuracy due to its ability to capture spatial features in the image data.

- [Week 2 Submission with thorough Analysis](https://drive.google.com/drive/folders/1FUs7ICKmWbXFGZthLeIAxzP9dQnqMPpV?usp=share_link)

### Week 3 - Reinforcement Learning (RL)

- Reinforcement Learning (RL) is a paradigm in machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward over time.

- Key Components - Agent, Environment, State, Action, Reward, Policy, Value Function, 

- Unlike supervised learning, RL doesn’t require labeled input-output pairs. Instead, it learns from trial and error

- Markov Decision Processes (MDPs) form the mathematical foundation for RL problems, describing states, actions, transitions, and rewards.

- Understanding the difference between model-free and model-based learning. while the former learns directly from experience (like Q-learning or SARSA), the latter involves planning using a known or learned model. 

- Exploration vs. Exploitation: Balancing greedy action selection vs. information-gathering actions.

- Understanding the need for Value Function Approximation - Tabular methods don’t scale; function approximation (e.g., linear, neural nets) is needed for large/continuous spaces.

- Policy Gradient Methods - Rather than learning a value function, directly optimize the policy via gradient ascent on expected return.

- [David Silver Lectures (classic series)](https://davidstarsilver.wordpress.com/teaching/)
