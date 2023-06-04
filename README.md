# Neuro-Gambit
## Overview
Neuro-Gambit is a Chess ANN experiment made and trained using pytorch.

The datasets for model were taken from the following links:
- https://www.ficsgames.org/download.html
- https://www.kaggle.com/datasets/datasnaek/chess
---
## The Models
There are two different models with different architectures trained for the purpose of exploring ANNs. Which are:

1. Neuro_Gambit: an ANN that simply contained linear layers with RelU activation functions.
2. Neuro_Gambit_resnet: a wrapper model around the resnet18 model. It contained a linear layer at the beginning and end to properly wrap the input and output of resnet.
---
## The inputs and outputs
### Input
The input to the models was essentially the board itself and the color of the player that the model is assuming. The board was linearized into `64` entries instead of the `8x8` matrix of chess. 

Furthermore, the `64` entries were one hot encoded to each have 13 possible values (corrosponding to various pieces of chess as well as an empty spot).

This makes the shape of the input to be `(64x13)+1=832+1=833`.

### Output
The output was organized into `36` labels which all corrospond to a specific next move in the board. The decoding is done as follows:
1. Seperate the `36` labels into `8x8x8x8x4`.
2. The first 8 labels corrospond into a one hot encoded origin rank movement in UCI notation.
3. The second 8 labels corrospond into a one hot encoded origin file movement in UCI notation.
4. The third and fourth label sets corropond to the one hot encoded destination rank and file movement in UCI notation.
5. The last 4 set corrosponds to the UCI promotion possibilites.

For example, if say the following tensor was the output:
```
[1., 0., 0., 0., 0., 0., 0., 0.] => a
[0., 0., 0., 0., 0., 0., 1., 0.] => 7
[1., 0., 0., 0., 0., 0., 0., 0.] => a
[0., 0., 0., 0., 0., 0., 0., 1.] => 8
        [1., 0., 0., 0.]         => q
```
As you can see, we can decode the output to be `a7a8q` which in UCI notation, implies that we are promoting the white `a7` pawn into a queen by moving it to `a8`. In algebraic notation, this would `a8=q`.

---
## Playing with the model
To play with the model, you will need to make sure you have the correct libraries installed. You can do so with the following command:
```
pip install -r requirements.txt
```

Afterwards, you can head into the `playing.ipynb` notebook and run the first 2 cells. You will be able to directly play with the AI in the notebook.

**Note** that you can only run this code on Linux as some libraries don't work on windows.

---
## Final thoughts
This project was an experiment and a way to learn how to utilize the pytorch library and how to make use of transfer learning. It was, however, a not exact in terms of the goal that we were trying to achieve. The models, regardless of how we tried, kept on overfitting where the cut out point seemed to always be with a `train loss =~ 0.3` and a `validation loss =~ 0.375`.

Any attempt of trying to avoid the overfitting seemed to fail and the most likely reason could have been that the input data shape itself did not warrent good generalization and that the general approach of seperating the moves from the game could remove some vital generalization.

The methods used to avoid overfitting are **L2 regularization**, **dropout neurons** and **early stopping**.

However, the model did seem to generalize a bit at the beginning of the game as it seemed to be defending and capturing correctly *(according to my judgement where I have an ELO of 1000 in chess.com)*. This could be because in the training cases, it simply would have seen more similar opennings as the number of openning is far less than the number of potential games of chess.

In conclusion, this model was able to generalize a bit when it came to the beginning of the game, always ended in overfitting which forced us to keep the training with around 0.3 loss cut point, and did not perfrom any better with he resnet model.

Thank you for your time :).