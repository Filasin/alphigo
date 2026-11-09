from alphigo.alphigo_board import _corner_tables


def is_point_an_eye(board, point, color):
    grid = board._grid
    if grid.get(point) is not None:
        return False
    for neighbor in board._neighbor_table[point]:
        s = grid.get(neighbor)
        if s is None or s.color != color:
            return False
    on_grid, off_board = _corner_tables[(board.num_rows, board.num_cols)][point]
    friendly = 0
    for corner in on_grid:
        s = grid.get(corner)
        if s is not None and s.color == color:
            friendly += 1
    if off_board > 0:
        return off_board + friendly == 4
    return friendly >= 3
