import numpy as np
import matplotlib.pyplot as plt
import os
import time

qTable = np.zeros((9, 9, 4))
gameSpace = [
    ['#','#','#','#','#','#','#','#','#'],
    ['#',' ',' ',' ',' ',' ',' ',' ','#'],
    ['#',' ',' ',' ',' ',' ',' ',' ','#'],
    ['#',' ',' ','#',' ','#',' ',' ','#'],
    ['#',' ',' ','#','O',' ',' ',' ','#'],
    ['#',' ',' ',' ','#','#',' ',' ','#'],
    ['#',' ',' ',' ','A',' ',' ',' ','#'],
    ['#',' ',' ',' ',' ',' ',' ',' ','#'],
    ['#','#','#','#','#','#','#','#','#']]

epsylon = 1
for _ in range(101):
    gameSpace[4][4] = 'O'
    quit = False
    r = 0
    epsylon -= 0.01
    pAx = cAx = 6
    pAy = cAy = 4
    while quit == False:
        if _ == 100:
            time.sleep(1)
        gameSpace[pAx][pAy] = ' '

        if epsylon >= np.random.randint(100)/100:
            a = np.random.randint(4)
        else:
            a = np.argmax(qTable[pAx][pAy])

        if a == 0 and gameSpace[pAx - 1][pAy] != '#':
            r = -.3
            cAx = pAx - 1
            cAy = pAy
        if a == 1 and gameSpace[pAx][pAy - 1] != '#':
            r = -.3
            cAx = pAx
            cAy = pAy - 1
        if a == 2 and gameSpace[pAx + 1][pAy] != '#':
            r = -.3
            cAx = pAx + 1
            cAy = pAy
        if a == 3 and gameSpace[pAx][pAy + 1] != '#':
            r = -.3
            cAx = pAx
            cAy = pAy + 1
        if a == 'q' or gameSpace[cAx][cAy] == 'O':
            r = 10
            quit = True
        if _ == 100:
            os.system("cls")
        gameSpace[cAx][cAy] = 'A'
        qTable[pAx][pAy][a] = qTable[pAx][pAy][a] + 0.5*(r + 0.99*np.max(qTable[cAx][cAy] - qTable[pAx][pAy][a]))
        if _ == 100:
            [print(*n) for n in qTable]
            [print(*n) for n in gameSpace]
            print('current', 'past')
            print([cAx, pAx], [pAx, pAy])

        pAx, pAy = cAx, cAy
