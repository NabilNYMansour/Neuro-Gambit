import torch
import torch.nn as nn
import chess
from chess_transformation import linear_to_matrix, matrix_to_board, board_to_matrix, matrix_to_linear, uci_to_alg
import torch.nn.functional as F
from torchvision import models

def one_hot_encode_labels(labels : list[str]):
    labels_dict = {'.': 0, 'K': 1, 'Q': 2, 'B': 3, 'N': 4, 'R': 5, 'P': 6, 'k': 7, 'q': 8, 'b': 9, 'n': 10, 'r': 11, 'p': 12}
    encoded_labels = torch.zeros(len(labels), len(labels_dict))
    for i, label in enumerate(labels):
        encoded_labels[i][labels_dict[label]] = 1
    return encoded_labels.view(-1, 64*13) # will return a 1x832 which is 64x13

def encode_player_col(player : str):
    player_col = torch.Tensor([1 if player == 'white' else 0])
    return player_col


# AIs
class Neuro_gambit(nn.Module):
    def __init__(self):
        super(Neuro_gambit, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(833, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 36)
        )

    def forward_str_input(self, positions : list[str], player : str):
        pos_tensor = one_hot_encode_labels(positions)
        player_col = encode_player_col(player)
        device = next(self.parameters()).device
        input_layer = torch.cat((pos_tensor, player_col.unsqueeze(1)), dim=1).to(device)
        return self.forward(input_layer)

    def forward(self, x : torch.Tensor):
        output = self.layers(x)
        return torch.split(output, [8, 8, 8, 8, 4], dim=1)
    
    def forward_chess_board_input(self, board : chess.Board, player : str):
        positions = matrix_to_linear(board_to_matrix(board))
        return self.forward_str_input(positions, player)

class Neuro_gambit_resnet(nn.Module):
    def __init__(self):
        super(Neuro_gambit_resnet, self).__init__()

        self.input_linear_layer = nn.Linear(833, 1 * 32 * 32)
        self.resnet_layer = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.output_linear_layer = nn.Linear(1000, 36)

        for param in self.resnet_layer.parameters():
            param.requires_grad = False

        self.resnet_layer.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_layer.conv1.requires_grad_(True)

    def forward_str_input(self, positions : list[str], player : str):
        pos_tensor = one_hot_encode_labels(positions)
        player_col = encode_player_col(player)
        device = next(self.parameters()).device
        input_layer = torch.cat((pos_tensor, player_col.unsqueeze(1)), dim=1).to(device)
        output = self.forward(input_layer)
        return output

    def forward(self, x : torch.Tensor):
        x = self.input_linear_layer(x)
        x = x.view(-1, 1, 32, 32)
        y = self.resnet_layer(x)
        y = self.output_linear_layer(y)
        return torch.split(y, [8, 8, 8, 8, 4], dim=1)

    def forward_chess_board_input(self, board : chess.Board, player : str):
        positions = matrix_to_linear(board_to_matrix(board))
        return self.forward_str_input(positions, player)


def get_all_possible_moves(board : chess.Board, player : str):
    possible_moves = []
    player_col = chess.WHITE if player == 'white' else chess.BLACK
    for move in board.legal_moves:
        if board.turn == player_col:
            uci_move = move.uci()
            possible_moves.append(uci_move)

    return possible_moves

def get_move_probability(move : str, tensor_tuple : tuple[torch.Tensor]):
    pos_rank_labels = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    pos_file_labels = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}

    cpu = torch.device('cpu')

    # restruct tensors to list matrix
    tensor_matrix = []
    for tensor in tensor_tuple[:4]:
        tensor_list = F.softmax(tensor, dim=0).to(cpu).tolist()
        tensor_matrix.append(tensor_list)

    # get indices
    o_r_i = pos_rank_labels[move[0]]
    o_f_i = pos_file_labels[move[1]]
    d_r_i = pos_rank_labels[move[2]]
    d_f_i = pos_file_labels[move[3]]

    # get probabilities
    p_o_r = tensor_matrix[0][o_r_i]
    p_o_f = tensor_matrix[1][o_f_i]
    p_d_r = tensor_matrix[2][d_r_i]
    p_d_f = tensor_matrix[3][d_f_i]

    return (p_o_r*p_o_f*p_d_r*p_d_f, [p_o_r,p_o_f,p_d_r,p_d_f])

def get_best_move(model : Neuro_gambit, board : list[str] | chess.Board, player : str):
    if type(board) != chess.Board:
        board : chess.Board = matrix_to_board(linear_to_matrix(board), player)
    possible_moves = get_all_possible_moves(board, player)
    with torch.no_grad():
        output_tensor = model.forward_chess_board_input(board, player)
    output_tensor_formatted = [t.view(-1,) for t in output_tensor]
    possible_moves_probabilities = {move : get_move_probability(move, output_tensor_formatted) for move in possible_moves}
    max_move_prob = {'move' : '', 'prob' : 0, 'prob_values' : []}
    for move in possible_moves_probabilities.keys():
        prob = possible_moves_probabilities[move][0]
        prob_values = possible_moves_probabilities[move][1]
        if prob > max_move_prob['prob']:
            max_move_prob['move'] = move
            max_move_prob['prob'] = prob
            max_move_prob['prob_values'] = prob_values

    if len(max_move_prob['move']) == 5: # there is a promotion
        promotion_prediction = output_tensor_formatted[4].softmax(0).tolist()
        index = promotion_prediction.index(max(promotion_prediction))
        promotion_labels = {0 : 'q', 1 : 'r', 2 : 'b', 3 : 'n'}
        max_move_prob['move'] = max_move_prob['move'][:4]+promotion_labels[index]

    return max_move_prob