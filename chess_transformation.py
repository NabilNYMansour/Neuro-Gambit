import chess

def algebraic_to_matrix_python_chess(moves : str):
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

def print_board(board : list[list[str]]):
    for row in board:
        print(row)

def board_to_linear(board : list[list[str]]):
    linear_board = []
    for row in board:
        for cell in row:
            linear_board.append(cell)
    return linear_board

def linear_to_board(board : list[str]):
    board_matrix = []
    for i in range(0, 64, 8):
        row = list(board[i:i+8])
        board_matrix.append(row)
    return board_matrix

def matrix_to_board(matrix : list[list[str]], player : str):
    board = chess.Board()
    board.clear() # clear the board
    for rank, row in enumerate(matrix):
        for file, piece_symbol in enumerate(row):
            if piece_symbol != '.':
                piece = chess.Piece.from_symbol(piece_symbol)
                square = chess.square(file, rank)
                board.set_piece_at(square, piece)
    board.turn = chess.WHITE if player == 'white' else chess.BLACK
    return board

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

if __name__ == '__main__':
    moves = 'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nc6 c4 e6 Nc3 Nf6 Be2 Be7 Be3 a6 O-O O-O Rc1 b5'
    # data = algebraic_game_to_training_dataset(moves, 'black')
    # print(moves)
    # print(data['moves_alg'])
    # print(data['moves_uci'])
    data = algebraic_game_to_training_dataset(moves, 'white')
    print(data['moves_alg'])
    print(data['moves_uci'])

    for i in range(len(data['boards'])):
        board = data['boards'][i]
        alg = data['moves_alg'][i]
        uci = data['moves_uci'][i]
        print('\n')
        print_board(linear_to_board(board))
        print(alg)
        print(uci)
