# Resources:

**Assignment Solutions:** https://drive.google.com/drive/folders/1G8THupL5Om1nP-mMofeJvexDy8UeA62-?usp=drive_link 

**Week1:**

https://www.w3schools.com/python/

https://www.tutorialspoint.com/python/python_oops_concepts.htm

https://www.geeksforgeeks.org/pygame-tutorial/

https://www.youtube.com/watch?v=QFvqStqPCRU

https://www.geeksforgeeks.org/snake-game-in-python-using-pygame-module/

Week2:

- For Neural Networks
    - 3blue1Brown (**all 4 videos**):   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    - Andrew NG. [Neural Networks by Andrew NG](https://youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)  (start from **video no. 25** )
    - Play with Neural Nets - https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.23879&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
    - Optional Videos for a deeper theoretical understanding
        - Lectures 4-6 by Andrej Karpathy (he has good videos covering many ML topics) [CS231n Winter 2016 - YouTube](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
        - Another good resource for ML: https://course.fast.ai/
- For CNNs
    - What is convolution? https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown
    - A video complete on its own: https://youtu.be/oGpzWAlP5p0?si=9nSI2P4u42y-S2ZA
    - Andrew NG: [CNN by Andrew NG](https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)  (**till video no. 13**)
    - Optional Videos for a deeper theoretical understanding
        - Lectures 6-8 by Andrej Karpathy:` [CS231n Winter 2016 - YouTube](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
- Pytorch Tutorials
    - Complete Tutorial: https://www.youtube.com/watch?v=OIenNRt2bjg&ab_channel=AssemblyAI
    - Another Tutorial covering many more topics (Optional and depends on your interest): https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
    - An image classifier built using Pytorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Transfer Learning (very popular with CNNs - Optional)
    - https://builtin.com/data-science/transfer-learning
    

Assignment: https://docs.google.com/document/d/1esRQ_lyeA5bLIfsktyW55RNGAS6Fc-BfrmtpHgyMDd8/edit?usp=sharing

Week3:

Complete all the lectures , book is also a good read but not mandatory

- [David Silver RL Lectures](https://www.davidsilver.uk/teaching/) (classic series) **OR** [2018 playlist (same content but not by David Silver)](https://youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb&si=DuKO-29ufPbOe0lR)
- [SpinningUp: RL Intr](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)o
- [Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

Assignment: No Assignment, but make notes for RL concepts so its easy to brush up the concepts

Week4:

Videos:

https://youtu.be/2GwBez0D20A?si=2jPdCaBkqt33n8zy

https://youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&si=x_jNPFube4pqQ6f2 

https://youtu.be/L8ypSXwyBds?si=kcl2AwPC_tz0hrxn

Reading: 

https://www.baeldung.com/cs/q-learning-vs-deep-q-learning-vs-deep-q-network 

https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9

Week5:

- Understanding Replay Buffer and Target Networks:
    - DQN using Atari Games: http://huggingface.co/learn/deep-rl-course/en/unit3/introduction
    - Medium article containing DQN code: https://medium.com/%40samina.amin/deep-q-learning-dqn-71c109586bae
    - https://www.geeksforgeeks.org/deep-q-learning/
    - Pytorch Implementation (Atari Game): [DQN — Stable Baselines3 2.7.0a0 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
    - For our theory enthusiasts (Optional):
        - https://arxiv.org/abs/1312.5602
    - More flavours of Q learning (Optional) - https://github.com/dxyang/DQN_pytorch
    - State of the Art DQNs (Optional) - https://github.com/higgsfield/RL-Adventure
- CNN as Q-function approximator
    - Approximators: https://gibberblot.github.io/rl-notes/single-agent/function-approximation.html
    - Conv-DQN: https://medium.com/%40hkabhi916/mastering-convolutional-deep-q-learning-with-pytorch-a-comprehensive-guide-0114742a0a62
- 

***Assignment:***

- Structure:
    - **Game Integration**
        - Start with your existing Pygame Snake code
        - Refactor as needed to match tutorial structure
    - **Part A – Q‑Learning Agent**
        - Follow Q-Learning section of the tutorials
        - Implement tabular Q-Learning with discrete state representation
        - Train and evaluate, tracking average score per episode
        - Resources you can refer to: [https://www.python-engineer.com/posts/teach-ai-snake-reinforcement-learning](https://www.python-engineer.com/posts/teach-ai-snake-reinforcement-learning/?utm_source=chatgpt.com)
    - **Part B – Deep Q‑Learning (DQN)**
        - Transition to neural network from tutorials
        - Implement replay buffer and target network
        - Train DQN, compare learning curves vs. Q‑Learning
        - Resources you can refer to: https://medium.com/%40nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c
    - Explore (Completely Optional):
        - Try using Conv-DQNs
        - Implement an **extension** like Double DQN
- Submission: Two different solutions - one for Q learning. one for Deep Q learning
A small report where you compare how both performed

Week6:

Reading:

https://notanymike.github.io/Solving-CarRacing/ 

https://openai.com/index/openai-baselines-ppo/

https://arxiv.org/html/2410.22766v1 

https://wandb.ai/mukilan/intro_to_gym/reports/A-Gentle-Introduction-to-OpenAI-Gym--VmlldzozMjg5MTA3

https://www.digitalocean.com/community/tutorials/getting-started-with-openai-gym 

Videos:

**PPO Algo**

https://youtu.be/5P7I-xPq8u8?si=AQnbEtdJFDI08nZ_

https://youtu.be/8jtAzxUwDj0?si=fClwjbx2k6lcN5HZ

https://youtu.be/MEt6rrxH8W4?si=-c_uCbJ5OqH6pbrW

https://youtu.be/05RMTj-2K_Y?si=i0WnvRnmjgBSlbFl 

https://youtu.be/BvZvx7ENZBw?si=qw0B5Ss8CfFIzJFv

https://youtu.be/HR8kQMTO8bk?si=a6Uk-vGCPMzH53AY 

**OpenAI Gym**

https://youtu.be/8MC3y7ASoPs?si=MG0dJXaKs9IBhehu ****

https://youtu.be/YLa_KkehvGw?si=Fyfy_dhWVbxNzseV 

# Final Project – PPO Agent for CarRacing

**Boilerplate**: https://colab.research.google.com/drive/1LUpZz__LoKeOkOXIjfM4LZBLNe1egduR

## Objective

Train a PPO  agent to solve the CarRacing-v0 environment using visual input. The goal is to develop an agent that can complete laps efficiently using image-based reinforcement learning.

---

## Tasks Breakdown

These are just a way to go through the project. You need not follow these strictly.

### 1. Environment Setup

- Install dependencies: gym, Box2D, opencv-python, numpy, torch, stable-baselines3, etc.
- Launch CarRacing-v3 (v2 and lower versions are deprecated) and explore the observation and action spaces.

### 2.  Image Preprocessing + Frame Stacking

- Convert RGB frames to grayscale (or keep RGB)
- Resize to a smaller resolution (e.g., 84x84 or 96x96)
- Normalize pixel values
- Stack 3–4 frames to give the agent temporal context

### 3. PPO Agent Implementation

- Use Stable-Baselines3 or implement PPO from scratch. We expect you to implement PPO from scratch and compare its performance with that of existing models in the report.
- Start with `CnnPolicy` and make sure it works end-to-end. You can use other policies too.
- Configure hyperparameters: learning rate, entropy coef, batch size, etc.
- Once it's working, explore enhancements:
    - Custom CNN architecture
    - Recurrent PPO with LSTM
    - Reward shaping

### 4. Reward Shaping and Evaluation

- (Optional) Modify the reward function to penalize off-track behavior
- Save checkpoints at regular intervals
- Evaluate the agent on unseen tracks
- Record gameplay using `gym.wrappers.RecordVideo` or `imageio`

---

## Resources

- PPO on CarRacing-v0 (elsheikh21):
    
    https://github.com/elsheikh21/car-racing-ppo
    
- Tutorial & theory:
    
    https://notanymike.github.io/Solving-CarRacing/
    
- Stable-Baselines3 Docs:
    
    https://stable-baselines3.readthedocs.io/en/master/
    
- PPO Paper (OpenAI):
    
    https://arxiv.org/abs/1707.06347
    
- Starter Notebook:
    
    PPO CarRacing Colab Starter (coming soon!)
    

---

## Final Deliverables

### 1. Trained PPO Agent

- Submit the final checkpoint of your trained agent
- Aim to achieve an average reward of *≥900* across 10 episodes (this is a typical benchmark from literature). However, focus more on learning and experimentation than hitting an exact score.

### 2. Report (PDF or Markdown)

Your report should briefly describe:

- Preprocessing pipeline (frame size, stack, grayscale?)
- PPO architecture & hyperparameters
- Training process (reward graphs, episodes)
- Reward shaping (if used)
- Observations: challenges, interesting behaviours, overfitting, generalization

### 3. Demo Video or GIF

- Record a short clip of your trained agent completing a track (30–60 seconds)

Please structure your submission with:

- `code/` folder containing the implementation
- `report.pdf` or `report.md`
- `media/` folder for any images or videos
- Any other files/folder you find relevant

---

## Optional Extensions (for curiosity)

- Use a custom CNN-based policy
- Add LSTM for memory (use RecurrentPPO)
- Try frame prediction or curiosity-based exploration
- Implement PPO from scratch
- Apply it to a different vision-based Gym environment (e.g., VizDoom)

---

## Timeline

You’ll have *2-3 weeks* to complete this project.

If you get stuck:

- Use the reference repos as guidance
- Ask questions in the group or in the weekly? meets

Happy racing 🚗💨
