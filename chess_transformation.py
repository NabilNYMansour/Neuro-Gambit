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
    for row in board[::-1]:
        print(row)

def board_to_linear(board : list[list[str]]):
    linear_board = []
    for row in board:
        for cell in row:
            linear_board.append(cell)
    return linear_board

# Testing
# moves_string = "e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc6 bxc6 Ra6 Nc4 a4 c3 a3 Nxa3 Rxa3 Rxa3 c4 dxc4 d5 cxd5 Qxd5 exd5 Be6 Ra8+ Ke7 Bc5+ Kf6 Bxf8 Kg6 Bxg7 Kxg7 dxe6 Kh6 exf7 Nf6 Rxh8 Nh5 Bxh5 Kg5 Rxh7 Kf5 Qf3+ Ke6 Bg4+ Kd6 Rh6+ Kc5 Qe3+ Kb5 c4+ Kb4 Qc3+ Ka4 Bd1#"
# moves_string = "e4 e5 Nf3 d6"# d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 Nf6 Bg5 O-O b5 Nc5 Bxf6 Bxf6 Bd3 Qd7 O-O Nxd3 Qxd3 c6 a4 cxd5 Nxd5 Qe6 Nc7 Qg4 Nxa8 Bd7 Nc7 Rc8 Nd5 Qg6 Nxf6+ Qxf6 Rfd1 Re8 Qxd6 Bg4 Qxf6 gxf6 Rd3 Bxf3 Rxf3 Rd8 Rxf6 Kg7 Rf3 Rd2 Rg3+ Kf8 c3 Re2 f3 Rc2 Rg5 f6 Rh5 Kg7 Rd1 Kg6 Rh3 Rxc3 Rd7 Rc1+ Kf2 Rc2+ Kg3 h5 Rxb7 Kg5 Rxa7 h4+ Rxh4 Rxg2+ Kxg2 Kxh4 b6 Kg5 b7 f5 exf5 Kxf5 b8=Q e4 Rf7+ Kg5 Qg8+ Kh6 Rh7#"
# board_matrix = algebraic_to_matrix_python_chess(moves_string)
# print_board(board_matrix)
# print(board_to_linear(board_matrix))
# NOTE: that top is white