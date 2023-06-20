# Written and completed by Team members of MGAA course hosted in LIACS 2023
# Modified and improved by R.Ma
# Version 1.0

import sys

import numpy as np
import spacy
import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
from gdpc import Block, Editor
from gdpc import __url__, geometry
from gdpc import geometry as geo
from gdpc.exceptions import InterfaceConnectionError, BuildAreaNotSetError
from perlin_noise import PerlinNoise
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import gc


eng = spacy.load("en_core_web_sm")  # Load the English model to tokenize English text

FILE_PATH1 = 'Data/train.csv'
FILE_PATH2 = 'Data/test.csv'
data_pipe1 = dp.iter.IterableWrapper([FILE_PATH1])
data_pipe1 = dp.iter.FileOpener(data_pipe1, mode='rb')
data_pipe1 = data_pipe1.parse_csv(skip_lines=1, delimiter=',', as_tuple=False)
data_pipe2 = dp.iter.IterableWrapper([FILE_PATH1])
data_pipe2 = dp.iter.FileOpener(data_pipe2, mode='rb')
data_pipe2 = data_pipe2.parse_csv(skip_lines=1, delimiter=',', as_tuple=False)


def engTokenize(text):
    """
    Tokenize an English text and return a list of tokens
    """
    return [token.text for token in eng.tokenizer(text)]

# print(engTokenize("Have a good day!!!"))

def getTokens(data_iter):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """
    for english in data_iter:
        yield engTokenize(english[0])


inputted = input('please input a piece of description of the world: ')
#print(type(inputted))
tokenized = engTokenize(inputted)
#print(tokenized)

vocab1 = build_vocab_from_iterator(
    getTokens(data_pipe1),  #### Refer to or.py
    specials=['<unk>'],
    special_first=True
)
vocab2 = build_vocab_from_iterator(
    getTokens(data_pipe2),  #### Refer to or.py
    specials=['<unk>'],
    special_first=True
)
vocab3 = build_vocab_from_iterator(
    getTokens(tokenized),  #### Refer to or.py
    specials=['<unk>'],
    special_first=True
)

vocab1.set_default_index(vocab1['<unk>'])
vocab2.set_default_index(vocab2['<unk>'])
vocab3.set_default_index(vocab3['<unk>'])


def getTransform(vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(1, begin=True),
        ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(2, begin=False)
    )

    return text_tranform


def applyTransform1(sequence_pair):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """
    return (
        getTransform(vocab1)(engTokenize(sequence_pair[0])),
        # getTransform(vocab1)(engTokenize(sequence_pair[1]))
        sequence_pair[1]
    )


def applyTransform2(sequence_pair):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """

    return (
        getTransform(vocab2)(engTokenize(sequence_pair[0])),
        # getTransform(vocab2)(engTokenize(sequence_pair[1]))
        sequence_pair[1]
    )


data_pipe1, data_pipe2 = data_pipe1.map(applyTransform1), data_pipe2.map(
    applyTransform2)  ## Apply the function to each element in the iterator
temp_list = list(data_pipe1)
#print(temp_list[0])


def applyTrans_pred(sequence):
    return (
        getTransform(vocab3)(engTokenize(sequence))
        # getTransform(vocab2)(engTokenize(sequence_pair[1]))
    )


def sortBucket(bucket):
    """
    Function to sort a given bucket. Here, we want to sort based on the length of
    source and target sequence.
    """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))


data_pipe1, data_pipe2 = data_pipe1.bucketbatch(
    batch_size=16, batch_num=5, bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sortBucket
), data_pipe2.bucketbatch(
    batch_size=16, batch_num=5, bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sortBucket
)

#print(list(data_pipe1)[0])


def separateSourceTarget(sequence_pairs):
    """
    input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
    """
    sources, targets = list(zip(*sequence_pairs))
    # print(sources, targets)
    targets = [int(t) for t in targets]
    sources = [tokens + ([0] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens
               in sources]
    # return sources,targets
    # print(sources, targets)
    return torch.tensor(sources, dtype=torch.int32), torch.tensor(targets)


## Apply the function to each element in the iterator
data_pipe1, data_pipe2 = data_pipe1.map(separateSourceTarget), data_pipe2.map(separateSourceTarget)
# print(list(data_pipe1)[0])


target_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

max_words = 25

from torch import nn

embed_len = 50
hidden_dim = 50
n_layers = 1


class RNNClassifier(nn.Module):
    def __init__(self):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=(len(vocab1) + len(vocab2)), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, len(target_classes))

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(n_layers, X_batch.size(0), hidden_dim))

        return self.linear(output[:, -1])

rnn_classifier = RNNClassifier()

for layer in rnn_classifier.children():
    print("Layer : {}".format(layer))
    print("Parameters : ")
    for param in layer.parameters():
        print(param.shape)
    print()


def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [], [], []
        for X, Y in val_loader:
            preds = model(X)

            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))


def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs + 1):
        losses = []
        for X, Y in tqdm(train_loader):
            # print(X,Y)
            Y_preds = model(X)
            # print('X is ',X)
            # print('Y_preds is ',Y_preds)
            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        # CalcValLossAndAccuracy(model, loss_fn, val_loader)


from torch.optim import Adam

epochs = 30
learning_rate = 1e-3

loss_fn = nn.CrossEntropyLoss()
rnn_classifier = RNNClassifier()
optimizer = Adam(rnn_classifier.parameters(), lr=learning_rate)




def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

    return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()


# add a input interface for the model that takes in a sentence and outputs the predicted class

import torch.nn.functional as F

# print(tokenized)
tokentoken = applyTrans_pred(inputted)
while len(tokentoken) < 25:
    tokentoken.append(0)

tokentensor = torch.tensor(tokentoken)
tokentensor = torch.unsqueeze(tokentensor, dim=0)
#print(tokentensor)
traintest = input("train or test? ")
if traintest == "test":
    rnn_classifier = torch.load('./rnn_classifier.pt')
    print("Model Loaded from working directory.")
else:
    TrainModel(rnn_classifier, loss_fn, optimizer, data_pipe1, data_pipe2, epochs)
    Y_actual, Y_preds = MakePredictions(rnn_classifier, data_pipe2)
    print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
    print("\nClassification Report : ")
    print(classification_report(Y_actual, Y_preds, target_names=target_classes))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_actual, Y_preds))
    torch.save(rnn_classifier, './rnn_classifier.pt')
    print("Model Saved at working directory.")
    rnn_classifier = RNNClassifier()

pred = rnn_classifier(tokentensor)
gc.collect()
probabilities = F.softmax(pred, dim=1).argmax(dim=1).detach().numpy()
# prob_reversed = probabilities
# predicted_class = torch.argmax(probabilities[0])

#print(tokentoken)
#print(len(tokentoken))
#print(probabilities)

editor = Editor()

try:
    editor.checkConnection()
except InterfaceConnectionError:
    print(
        f"Error: Could not connect to the GDMC HTTP interface at {editor.host}!\n"
        "To use GDPC, you need to use a \"backend\" that provides the GDMC HTTP interface.\n"
        "For example, by running Minecraft with the GDMC HTTP mod installed.\n"
        f"See {__url__}/README.md for more information."
    )
    sys.exit(1)

try:
    buildArea = editor.getBuildArea()
except BuildAreaNotSetError:
    print(
        "Error: failed to get the build area!\n"
        "Make sure to set the build area with the /setbuildarea command in-game.\n"
        "For example: /setbuildarea ~0 0 ~0 ~64 200 ~64"
    )
    sys.exit(1)


# if any(dropY(buildArea.size) < MIN_BUILD_AREA_SIZE):
#     print(
#         "Error: the build area is too small for this example!\n"
#         f"It should be at least {tuple(MIN_BUILD_AREA_SIZE)} blocks large in the XZ-plane."
#     )
#     sys.exit(1)


def clear(size,x_build, y_build, z_build, height=50):
    geometry.placeCuboid(editor, (x_build, y_build , z_build),
                         (x_build+size, y_build+height , z_build+size), Block("air"))


def build(size, x_build, y_build, z_build, eles, blocktypes, terrain):
    s = int(size * 0.5)
    for i in range(s):
        for j in range(s):
            editor.placeBlock((x_build + i, y_build + eles[0][i][j], z_build + j), Block(blocktypes[0]))
            editor.placeBlock((x_build + s + i, y_build + eles[1][i][j], z_build + j), Block(blocktypes[1]))
            editor.placeBlock((x_build + i, y_build + eles[2][i][j], z_build + s + j), Block(blocktypes[2]))
            editor.placeBlock((x_build + s + i, y_build + eles[3][i][j], z_build + s + j), Block(blocktypes[3]))
            if terrain != 'Desert':

                geo.placeCuboid(editor, (x_build + i, y_build + eles[0][i][j]-1, z_build + j),
                            (x_build + i, y_build, z_build + j), Block("grass_block"))
                geo.placeCuboid(editor, (x_build + s + i, y_build + eles[1][i][j]-1, z_build + j),
                            (x_build + s + i, y_build, z_build + j), Block("grass_block"))
                geo.placeCuboid(editor, (x_build + i, y_build + eles[2][i][j]-1, z_build + s + j),
                            (x_build + i, y_build, z_build + s + j), Block("stone"))
                geo.placeCuboid(editor, (x_build + s + i, y_build + eles[3][i][j]-1, z_build + s + j),
                            (x_build + s + i, y_build, z_build + s + j), Block("stone"))
            else:
                geo.placeCuboid(editor, (x_build + i, y_build + eles[0][i][j] - 1, z_build + j),
                                (x_build + i, y_build, z_build + j), Block("sand"))
                geo.placeCuboid(editor, (x_build + s + i, y_build + eles[1][i][j] - 1, z_build + j),
                                (x_build + s + i, y_build, z_build + j), Block("sand"))
                geo.placeCuboid(editor, (x_build + i, y_build + eles[2][i][j] - 1, z_build + s + j),
                                (x_build + i, y_build, z_build + s + j), Block("sand"))
                geo.placeCuboid(editor, (x_build + s + i, y_build + eles[3][i][j] - 1, z_build + s + j),
                                (x_build + s + i, y_build, z_build + s + j), Block("sand"))


def plant_tree(size, STARTX, STARTY, STARTZ, eles, terrain, times):
    if terrain == 'Temperate Forest':
        tree_type = ['oak_sapling', 'spruce_sapling', 'red_tulip', 'red_mushroom', 'bamboo']
    elif terrain == 'Boreal Forest':
        tree_type = ['dark_oak_sapling', 'spruce_sapling', 'blue_orchid', 'fern', 'sweet_berry_bush']
    elif terrain == 'Tundra':
        tree_type = ['spruce_sapling', 'dead_bush', 'white_tulip', 'fern', 'sweet_berry_bush']
    elif terrain == 'Temperate Rainforest ':
        tree_type = ['oak_sapling', 'spruce_sapling', 'allium', 'brown_mushroom', 'sweet_berry_bush']
    elif terrain == 'Savanna':
        tree_type = ['acacia_sapling', 'oak_sapling', 'sunflower', 'tall_grass', 'sweet_berry_bush']
    elif terrain == 'Rainforest':
        tree_type = ['jungle_sapling', 'oak_sapling', 'brown_mushroom', 'sweet_berry_bush']
    elif terrain == 'Seasonal Forest':
        tree_type = ['oak_sapling', 'birch_sapling', 'dandelion', 'fern', 'sweet_berry_bush']
    elif terrain == 'Tropical Woodland':
        tree_type = ['jungle_sapling', 'oak_sapling', 'azure_bluet', 'brown_mushroom', 'sweet_berry_bush']
    elif terrain == 'Desert':
        tree_type = ['acacia_sapling', 'dead_bush']
    else:
        tree_type = ['dirt']

    s = int(size * 0.5)
    for i in range(times):
        x = np.random.randint(s)
        y = np.random.randint(s)
        plantType = np.random.choice(tree_type)
        # print(plantType)
        editor.placeBlock((STARTX + x, STARTY + eles[0][x][y] , STARTZ + y), Block(np.random.choice(tree_type)))
        editor.placeBlock((STARTX + s + x, STARTY + eles[1][x][y] + 1, STARTZ + y), Block(np.random.choice(tree_type)))
        editor.placeBlock((STARTX + x, STARTY + eles[2][x][y] + 1, STARTZ + s + y), Block(np.random.choice(tree_type)))
        editor.placeBlock((STARTX + s + x, STARTY + eles[3][x][y] + 1, STARTZ + s + y), Block(np.random.choice(tree_type)))


def create_map(blockType, tree_numbers, terrain, size, alpha=1):
    noise = PerlinNoise(octaves=3)
    size = size
    tree_numbers = 100
    terrain = terrain
    pic = [[noise([i / size, j / size]) for j in range(size)] for i in range(size)]
    # altitude alpha
    alpha = alpha
    ele = np.array(pic)
    ele = alpha * ele
    ele = np.round(ele)
    mini = ele.min()
    # print(mini)
    ele = ele - mini
    return ele


def main():
    # input parameters which should be trained by neural network
    terrainType = 'plain'
    terrain_list = ['Temperate Forest', 'Boreal Forest', 'Tundra', 'Temperate Rainforest ', 'Savanna',
                    'Rainforest', 'Seasonal Forest', 'Tropical Woodland', 'Desert']
    terrain_index = int(probabilities.item())
    #print('index is: ', terrain_index)
    #index_reversed = 8 - terrain_index
    #print('index reversed is: ', index_reversed)
    terrain = terrain_list[terrain_index]
    #terrain = 'Desert'
    print('The selected terrain by the neural network is: ',terrain)
    size = 100
    # the numbers of plants
    numbers = 100
    times = np.random.randint(100, 200)
    # control the flatness of not as small as flat
    #
    # if terrainType == 'plain':
    #     create_map('grass_block', numbers, terrain, size, alpha=np.random.choice([2,3]))
    # elif terrainType == 'hills':
    #     create_map('stone', numbers, terrain,  size, alpha=np.random.choice([6,7,8]))
    # elif terrainType == 'valley':
    #     create_map('stone',  numbers, terrain, size, alpha=np.random.choice([10,13,15]))
    # elif terrainType == 'mountainous':
    #     create_map('stone', numbers, terrain, size, alpha=np.random.choice([20,25,30]))

    BUILD_AREA = editor.getBuildArea()
    STARTX, STARTY, STARTZ = BUILD_AREA.begin
    ENDX, ENDY, ENDZ = BUILD_AREA.end

    # print(STARTX, STARTY, STARTZ, ENDX, ENDY, ENDZ)
    clear(size,STARTX, STARTY, STARTZ)
    if terrain != 'Desert':
        ele1 = create_map('grass_block', numbers, terrain, int(size/2), alpha=np.random.choice([3, 4, 5]))
        ele2 = create_map('stone', numbers, terrain,int(size/2), alpha=np.random.choice([7, 8, 9]))
        ele3 = create_map('stone', numbers, terrain, int(size/2), alpha=np.random.choice([13, 14, 15]))
        ele4 = create_map('stone', numbers, terrain, int(size/2), alpha=np.random.choice([24, 25, 26]))
        eles = [ele1, ele2, ele3, ele4]
        blocktypes = ['air', 'grass_block', 'grass_block', 'grass_block']
    else:
        ele1 = create_map('sand', numbers, terrain, int(size/2), alpha=np.random.choice([3, 4, 5]))
        ele2 = create_map('sand', numbers, terrain, int(size/2), alpha=np.random.choice([7, 8, 9]))
        ele3 = create_map('sand', numbers, terrain, int(size/2), alpha=np.random.choice([13, 14, 15]))
        ele4 = create_map('sand', numbers, terrain, int(size/2), alpha=np.random.choice([24, 25, 26]))
        eles = [ele1, ele2, ele3, ele4]
        blocktypes = ['sand', 'sand', 'sand', 'sand']


    print('building terrain in progress.')
    build(size, STARTX, STARTY, STARTZ,eles,blocktypes, terrain)
    print('Planting vegetation in progress..')
    plant_tree(size, STARTX, STARTY, STARTZ, eles, terrain, times)
    print('building completed.')


if __name__ == '__main__':
    main()
