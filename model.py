import torch
import torch.nn as nn
import chess
from chess_transformation import board_to_matrix, matrix_to_linear

def one_hot_encode_labels(labels : list[str]):
    labels_dict = {'.': 0, 'K': 1, 'Q': 2, 'B': 3, 'N': 4, 'R': 5, 'P': 6, 'k': 7, 'q': 8, 'b': 9, 'n': 10, 'r': 11, 'p': 12}
    encoded_labels = torch.zeros(len(labels), len(labels_dict))
    for i, label in enumerate(labels):
        encoded_labels[i][labels_dict[label]] = 1
    return encoded_labels.view(-1, 64*13) # will return a 1x832 which is 64x13

def encode_player_col(player : str):
    player_col = torch.Tensor([1 if player == 'white' else 0])
    return player_col

class Neuro_gambit(nn.Module):
    def __init__(self):
        super(Neuro_gambit, self).__init__()
        # self.lin_test = nn.Linear(833, 36) # input 64*13 + 1, output 36
        self.encoder = nn.Sequential(
            nn.Linear(833, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36)
        )

    def forward_str_input(self, positions : list[str], player : str):
        pos_tensor = one_hot_encode_labels(positions)
        player_col = encode_player_col(player)
        device = next(self.parameters()).device
        input_layer = torch.cat((pos_tensor, player_col.unsqueeze(1)), dim=1).to(device)
        output = self.decoder(self.encoder(input_layer))
        return torch.split(output, [8, 8, 8, 8, 4], dim=1)

    def forward(self, x : torch.Tensor):
        output = self.decoder(self.encoder(x))
        return torch.split(output, [8, 8, 8, 8, 4], dim=1)
    
    def forward_chess_board_input(self, board : chess.Board, player : str):
        positions = matrix_to_linear(board_to_matrix(board))
        return self.forward_str_input(positions, player)