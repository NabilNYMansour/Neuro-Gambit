import chess

def algebraic_to_matrix_python_chess(moves_string):
    moves = moves_string.split()
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

moves_string = "e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc6 bxc6 Ra6 Nc4 a4 c3 a3 Nxa3 Rxa3 Rxa3 c4 dxc4 d5 cxd5 Qxd5 exd5 Be6 Ra8+ Ke7 Bc5+ Kf6 Bxf8 Kg6 Bxg7 Kxg7 dxe6 Kh6 exf7 Nf6 Rxh8 Nh5 Bxh5 Kg5 Rxh7 Kf5 Qf3+ Ke6 Bg4+ Kd6 Rh6+ Kc5 Qe3+ Kb5 c4+ Kb4 Qc3+ Ka4 Bd1#"
print(moves_string)
board_matrix = algebraic_to_matrix_python_chess(moves_string)

for row in board_matrix:
    print(row)

