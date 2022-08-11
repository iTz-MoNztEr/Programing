from msvcrt import getch
import threading
import time
import os
import random


a = 119
def game_loop():
    eaten = True
    hr, hc = 5, 5
    snake = [60, 71, 82, 93]
    while True:
        if eaten == True:
            apple_r = random.randint(1, 9)
            apple_c = random.randint(1, 9)
            eaten = False
        os.system("cls")
        game_state = [
        'w','w','w','w','w','w','w','w','w','w','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w',' ',' ',' ',' ',' ',' ',' ',' ',' ','w',
        'w','w','w','w','w','w','w','w','w','w','w']
                                                                                #print(snake)
        if a == 119:
            hr -= 1
        if a == 97:
            hc -= 1
        if a == 115:
            hr += 1
        if a == 100:
            hc += 1
        if a == 113:
            quit()

        for n in range(len(snake)):
            if len(snake)-2-n >= 0:
                snake[len(snake)-1-n] = snake[len(snake)-2 - n]
        head = hr*11+hc
        snake[0] = head
                                                                                #print(snake)
        if len(snake) == 20:
            quit()
        if snake[0] == apple_r*11+apple_c:
            eaten = True
            snake.append(snake[-1] + 11)
        if snake[0] in snake[1:]:
            quit()
        if snake[0] in [i for i in range(11)]\
        or snake[0] in [len(game_state)-i for i in range(12)]\
        or snake[0] in [i*11 for i in range(11)]\
        or snake[0] in [i*11+10 for i in range(11)]:
            quit()

        game_state[apple_r*11+apple_c] = 'A'
        for n in snake:
            game_state[n] = 'B'
        game_state[head] = 'H'
        for i in range(len(game_state)):
            print(game_state[i], end = ' ')
            if i%11 == 10:
                print()
        time.sleep(.2)

loop_thread = threading.Thread(target = game_loop)
loop_thread.start()
while True:
    a = ord(getch())
loop_thread.join()
