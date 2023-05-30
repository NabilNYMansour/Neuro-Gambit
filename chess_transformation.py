import chess
import torch

# NOTE: Possible issue in board matrix order with some functions and the naming confuses board with matrix sometimes.

def algebraic_to_matrix(moves : str):
    if type(moves) == str:
        moves = moves.split()
    board = chess.Board()

    for move in moves:
        try:
            board.push_san(move)
        except ValueError:
            print(f"Invalid move: {move}")

    board_matrix = [['.' for _ in range(8)] for _ in range(8)]

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file, rank = chess.square_file(square), chess.square_rank(square)
            board_matrix[rank][file] = piece.symbol()

    return board_matrix

def print_matrix(board : list[list[str]]):
    for row in board:
        print(row)

def matrix_to_linear(board : list[list[str]]):
    linear_board = []
    for row in board:
        for cell in row:
            linear_board.append(cell)
    return linear_board

def linear_to_matrix(board : list[str]):
    board_matrix = []
    for i in range(0, 64, 8):
        row = list(board[i:i+8])
        board_matrix.append(row)
    return board_matrix

def matrix_to_board(matrix : list[list[str]], player : str):
    board = chess.Board()
    board.clear() # clear the board
    for rank, row in enumerate(matrix[::-1]):
        for file, piece_symbol in enumerate(row):
            if piece_symbol != '.':
                piece = chess.Piece.from_symbol(piece_symbol)
                square = chess.square(file, rank)
                board.set_piece_at(square, piece)
    board.turn = chess.WHITE if player == 'white' else chess.BLACK
    return board

def board_to_matrix(board : chess.Board):
    board_matrix = [['.' for _ in range(8)] for _ in range(8)]

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file, rank = chess.square_file(square), chess.square_rank(square)
            board_matrix[rank][file] = piece.symbol()

    return board_matrix[::-1]

def algebraic_to_full_move(board_matrix : list[list[str]], move : str, player : str):
    board = matrix_to_board(board_matrix, player)
    move_obj = board.parse_san(move)  # Parse the move in algebraic notation
    return str(move_obj)

def full_move_to_algebraic(board_matrix : list[list[str]], full_move : str, player : str):
    move_obj = chess.Move.from_uci(full_move)  # Convert the first 4 characters to a Move object
    board = matrix_to_board(board_matrix, player)
    algebraic_move = board.san(move_obj)  # Convert the Move object to algebraic notation
    return algebraic_move

def algebraic_game_to_training_dataset(moves: str, winner : str):
    if type(moves) == str:
        moves = moves.split()
    board = chess.Board()  # board init
    data = {'boards' : [], 'moves_alg' : [], 'moves_uci' : []}
    board_linear = list(str(board).replace('\n',' ').replace(' ', ''))
    if winner == 'white':
        data['boards'].append(board_linear)
        data['moves_alg'].append(moves[0])
        data['moves_uci'].append(str(board.parse_san(moves[0])))
    for i in range(len(moves)-1):
        move = moves[i]
        n_move = moves[i+1]
        try:
            # print(board)
            # print('\n')
            board.push_san(move)
            n_move_uci = str(board.parse_san(n_move))
            board_linear = list(str(board).replace('\n',' ').replace(' ', ''))
            if winner == 'white' and i%2==1:
                data['boards'].append(board_linear)
                data['moves_alg'].append(n_move)
                data['moves_uci'].append(n_move_uci)
            elif winner == 'black' and i%2==0:
                data['boards'].append(board_linear)
                data['moves_alg'].append(n_move)
                data['moves_uci'].append(n_move_uci)
        except ValueError:
            print(f"Invalid move: {move}")

    return data

def encode_uci(uci : str):
    pos_rank_labels = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    pos_file_labels = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
    promotion_labels = {'q': 0, 'r': 1, 'b': 2, 'n': 3}

    origin_pos = uci[:2]
    destin_pos = uci[2:4]

    org_rank = origin_pos[0]
    org_file = origin_pos[1]

    des_rank = destin_pos[0]
    des_file = destin_pos[1]

    # org pos
    encoded_org_pos_rank = torch.zeros(1, len(pos_rank_labels))
    encoded_org_pos_rank[0][pos_rank_labels[org_rank]] = 1

    encoded_org_pos_file = torch.zeros(1, len(pos_file_labels))
    encoded_org_pos_file[0][pos_file_labels[org_file]] = 1

    # des pos
    encoded_des_pos_rank = torch.zeros(1, len(pos_rank_labels))
    encoded_des_pos_rank[0][pos_rank_labels[des_rank]] = 1

    encoded_des_pos_file = torch.zeros(1, len(pos_file_labels))
    encoded_des_pos_file[0][pos_file_labels[des_file]] = 1

    encoded_prom = torch.zeros(1, len(promotion_labels)) # all zeros

    if (len(uci) == 5): # there is a promotion
        promotion = uci[4]
        for i, label in enumerate(promotion):
            encoded_prom[i][promotion_labels[label]] = 1

    return torch.cat((encoded_org_pos_rank.view(1, -1), encoded_org_pos_file.view(1, -1),
                      encoded_des_pos_rank.view(1, -1), encoded_des_pos_file.view(1, -1),
                      encoded_prom.view(1, -1)), dim=1)

def decode_uci(tensors : list[torch.Tensor]):
    pos_rank_labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    pos_file_labels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}
    promotion_labels = {0: 'q', 1: 'r', 2: 'b', 3: 'n'}

    tensor_list = []
    for i in range(len(tensors)):
        tensor_list.append(tensors[i].tolist()[0])
    o_r = pos_rank_labels[tensor_list[0].index(1)]
    o_f = pos_file_labels[tensor_list[1].index(1)]
    d_r = pos_rank_labels[tensor_list[2].index(1)]
    d_f = pos_file_labels[tensor_list[3].index(1)]
    try:
        p = promotion_labels[tensor_list[4].index(1)]
    except ValueError:
        p = ''
    return o_r+o_f+d_r+d_f+p

if __name__ == '__main__':
    moves = 'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nc6 c4 e6 Nc3 Nf6 Be2 Be7 Be3 a6 O-O O-O Rc1 b5'
    data = algebraic_game_to_training_dataset(moves, 'black')
    b = linear_to_matrix(data['boards'][0])
    print_matrix(b)
    b = matrix_to_board(b, 'black')
    print(b)
    b = board_to_matrix(b)
    print_matrix(b)
    b = linear_to_matrix(matrix_to_linear(b))
    print_matrix(b)
    b = board_to_matrix(matrix_to_board(linear_to_matrix(matrix_to_linear(b)), 'black'))
    print_matrix(b)
    # print(data['moves_alg'])
    # print(data['moves_uci'])
    # data = algebraic_game_to_training_dataset(moves, 'white')
    # print(data['moves_alg'])
    # print(data['moves_uci'])

    # for i in range(len(data['boards'])):
    #     board = data['boards'][i]
    #     alg = data['moves_alg'][i]
    #     uci = data['moves_uci'][i]
    #     print('\n')
    #     print_matrix(linear_to_matrix(board))
    #     print(alg)
    #     print(uci)
    # b = chess.Board()
    # print(b)
    # print()

    # bm = board_to_matrix(b)
    # print_matrix(bm)
    # print()

    # m_to_b = matrix_to_board(bm, 'white')
    # print(m_to_b)
    # pass
