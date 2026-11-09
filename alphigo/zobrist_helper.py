import random

from alphigo.alphigo_types import Player, Point


def to_python(player_state):
    if player_state is None or player_state == 'None':
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white


MAX63 = 0x7fffffffffffffff

random.seed(42)

table = {}

empty_board = random.randint(0, MAX63)

for row in range(1, 20):
    for col in range(1, 20):
        for state in ('None', Player.black, Player.white):
            code = random.randint(0, MAX63)
            table[Point(row=row, col=col), state] = code

print('from .alphigo_types import Player, Point')
print('')
print('HASH_CODE = {')
for (pt, state), hash_code in table.items():
    print(' (%r, %s): %r,' % (pt, to_python(state), hash_code))
print('}')
print('')
print('EMPTY_BOARD = %d' % (empty_board,))
