# -*- coding: utf-8 -*-
import pong_environment
import random
import pygame

class PongGame:
    def __init__(self):
        """
        1. Paddle 1:
            a. Paddle 1’s Y position
        2. Paddle 2:
            a. Paddle 2’s Y position
        3. Tally : ​ To keep track of the score
        4. Ball :
            a. Position along X axis
            b. Position along Y axis
            c. Direction along the X Axis
            d. Direction along the Y Axis
        """
        self.env = pong_environment.pong_environment()
        pygame.font.init()
        num = random.randint(0,9)
        self.tally = 0
        self.paddle1YPos = self.env.WINDOW_HEIGHT / 2 - self.env.PADDLE_HEIGHT / 2
        self.paddle2YPos = self.env.WINDOW_HEIGHT / 2 - self.env.PADDLE_HEIGHT / 2
        self.ballXDirection = 1
        self.ballYDirection = 1
        self.ballXPos = self.env.WINDOW_WIDTH/2 - self.env.BALL_WIDTH/2

        if(0 < num < 3):
            self.ballXDirection = 1
            self.ballYDirection = 1
        if (3 <= num < 5):
            self.ballXDirection = -1
            self.ballYDirection = 1
        if (5 <= num < 8):
            self.ballXDirection = 1
            self.ballYDirection = -1
        if (8 <= num < 10):
            self.ballXDirection = -1
            self.ballYDirection = -1

        num = random.randint(0,9)
        self.ballYPos = num*(self.env.WINDOW_HEIGHT - self.env.BALL_HEIGHT)/9


    """
    The ​ controller class can interact with the game class, through two specific methods​ :
    1. FirstFrame :
        a. Accepts No Arguments
        b. Returns image data from the first frame
    2. NextFrame :
        a. Accepts the action taken by the agent
        b. Returns the image data from the next frame and the score of the game.
    """
    def getPresentFrame(self):
        pygame.event.pump()
        self.env.screen.fill(self.env.BLACK)
        self.env.drawPaddle1(self.paddle1YPos)
        self.env.drawPaddle2(self.paddle2YPos)
        self.env.drawBall(self.ballXPos, self.ballYPos)
        self.env.drawScore(self.tally)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return image_data


    def getNextFrame(self, action, infos):
        pygame.event.pump()
        score               = 0
        self.env.screen.fill(self.env.BLACK)
        self.paddle1YPos    = self.env.updatePaddle1(action, self.paddle1YPos)
        self.env.drawPaddle1(self.paddle1YPos)
        self.paddle2YPos    = self.env.updatePaddle2(self.paddle2YPos, self.ballYPos)
        self.env.drawPaddle2(self.paddle2YPos)
        [score, self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = self.env.updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection)
        self.env.drawBall(self.ballXPos, self.ballYPos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.env.drawScore(self.tally)
        self.env.drawInfos(infos, action)
        pygame.display.flip()
        self.tally = self.tally + score
        if self.tally==-21:
            self.tally=0
        return [score, image_data]
