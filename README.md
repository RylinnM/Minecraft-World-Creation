# Minecraft-World-Creation
A Minecraft game plugin that can generate different terrains according to user descriptions, combining PCG algorithm and NLP.

## Scope
The idea behind this project is the creation of a game world in Minecraft that functions like a black box. It is by definition created by a seed, which is an integer number. For example, "seed 42". However, this process brings uncertainty and misunderstanding of how the world actually is, as people can not imagine a world just with a number, even for hardcore players. Therefore, a plugin can take in the descriptive sentences input by players and then generate a game world accordingly. This will make the entire process more understandable and transparent.

## Description
This Python project, developed by the team members of the MGAA course at LIACS in 2023 and further refined by R. Ma, utilizes a variety of libraries including Numpy, Spacy, PyTorch, and GDPC to perform natural language processing and terrain generation in a simulated environment. The code is designed to tokenize English text, build vocabularies from datasets, transform sequences, and classify them using a RNN (Recurrent Neural Network). It also interfaces with the Minecraft game to create terrain based on the classifications made by the RNN.

## Version
1.0

## Key Features
- **English Text Tokenization**: Tokenizes English sentences using Spacy's NLP model.
- **Vocabulary Building**: Constructs vocabularies from given text data.
- **Sequence Transformation**: Applies transformations to sequences of tokens.
- **RNN Classification**: Uses a RNN model for classifying sequences into different categories.
- **Terrain Generation**: Interfaces with Minecraft to generate terrain based on classification results.
- **Plant and Tree Planting**: Depending on the classified terrain type, different plants and trees are planted.

## Prerequisites
- Python 3.x
- PyTorch
- TorchText
- Spacy
- GDPC (with Minecraft and GDMC HTTP interface)
- NumPy
- Scikit-Learn
- tqdm
- A working Minecraft setup with the GDMC HTTP mod installed.

## Installation
1. Clone the repository or download the source code.
2. Install required Python packages: `pip install numpy spacy torch torchtext sklearn tqdm gdpc`
3. Install Spacy's English model: `python -m spacy download en_core_web_sm`

## Usage
1. Run the script using Python.
2. Start running the Minecraft game and open a super flat world.
3. Input a description of the world when prompted.
4. The script will interact with Minecraft to generate terrain based on the classification.

## Contact Information
For further inquiries, please contact the repo owner.

## Disclaimer
This project is part of an academic course and is intended for educational purposes only. It requires a specific setup with Minecraft and GDPC HTTP interface for full functionality.

