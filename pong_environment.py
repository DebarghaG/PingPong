# -*- coding: utf-8 -*-
import pygame
import random
import os

"""
For training inside the Google Collab :

#os.environ['SDL_VIDEODRIVER']='dummy'
"""

class pong_environment:
    def __init__(self):

        """
        Main Subsections :
        1. Paddle 1 : ​ Controlled by the bot or the human playing, which has :
            a. Paddle Height
            b. Paddle Width
            c. Paddle Buffer : Space between the edge of the window and the paddle.

        2. Paddle 2 :​ Controlled by the agent that learns.
            a. Paddle Height
            b. Paddle Width
            c. Paddle Buffer : Space between the edge of the window and the paddle.

        3. Ball : ​ Makes perfectly elastic collisions against the walls, as well as the agent.
            a. Ball Width
            b. Ball Height
            c. Ball X Velocity
            d. Ball Y Velocity
        """

        self.WINDOW_WIDTH   = 400
        self.WINDOW_HEIGHT  = 400
        self.PADDLE_WIDTH   = 10
        self.PADDLE_HEIGHT  = 60
        self.PADDLE_BUFFER  = 10

        self.BALL_WIDTH     = 10
        self.BALL_HEIGHT    = 10


        self.PADDLE_SPEED   = 5
        self.BALL_VelX      = 5
        self.BALL_VelY      = 5

        self.BLACK          = (0, 0, 0)
        self.WHITE          = (255, 255, 255)
        """
        self.BLACK = (255,255,255)
        self.WHITE = (0,0,0)
        """

        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))


    def drawBall(self,ballXPos, ballYPos):
        ball = pygame.Rect(ballXPos, ballYPos, self.BALL_WIDTH, self.BALL_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, ball)


    def drawPaddle1(self,paddle1YPos):
        paddle1 = pygame.Rect(self.PADDLE_BUFFER, paddle1YPos, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, paddle1)


    def drawPaddle2(self, paddle2YPos):
        paddle2 = pygame.Rect(self.WINDOW_WIDTH - self.PADDLE_BUFFER - self.PADDLE_WIDTH, paddle2YPos, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, paddle2)

    def drawScore(self,score):
        obsolete_function=""
        """
        font = pygame.font.Font(None, 20)
        scorelabel = font.render("Score " + str(score), 1, self.WHITE)
        self.screen.blit(scorelabel, (60 , 10))
        """

    def drawInfos(self,infos, action):

        """
        Commented out because not necessary since we've eliminated the headers


        font        = pygame.font.Font(None, 20)
        label       = font.render("step " + str(infos[0]) + " ["+str(infos[3])+"]", 1, self.WHITE)
        self.screen.blit(label, (60 , 30))
        label       = font.render("epsilon " + str(infos[2]), 1, self.WHITE)
        self.screen.blit(label, (30 , 45))
        label       = font.render("q_max " + str(infos[1]), 1, self.WHITE)
        self.screen.blit(label, (30 , 60))
        actionText  = "--"
        if (action[1] == 1):
            actionText = "Up"
        if (action[2] == 1):
            actionText = "Down"
        label       = font.render("Action " + actionText, 1, self.WHITE)
        self.screen.blit(label, (60 , 40))
        """


    def updateBall(self,paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection):

        ballXPos     = ballXPos + ballXDirection * self.BALL_VelX
        ballYPos     = ballYPos + ballYDirection * self.BALL_VelY
        score        = 0

        if (ballXPos <= self.PADDLE_BUFFER + self.PADDLE_WIDTH and ballYPos + self.BALL_HEIGHT >= paddle1YPos and ballYPos - self.BALL_HEIGHT <= paddle1YPos + self.PADDLE_HEIGHT):
            ballXDirection  = 1
            score           = 1

        elif (ballXPos <= 0):
            ballXDirection  = 1
            score           =-1
            return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

        if (ballXPos >= self.WINDOW_WIDTH - self.PADDLE_WIDTH - self.PADDLE_BUFFER and ballYPos + self.BALL_HEIGHT >= paddle2YPos and ballYPos - self.BALL_HEIGHT <= paddle2YPos + self.PADDLE_HEIGHT):
            """
            For Adding noise to our simulations :

            x = random.uniform(-0.5,0.5)
            ballXDirection = -1+x
            x = random.uniform(-0.5,0.5)
            self.BALL_VelX = self.BALL_VelX+x
            """
            ballXDirection = -1

        elif (ballXPos >= self.WINDOW_WIDTH - self.BALL_WIDTH):
            ballXDirection = -1
            score          = 1
            return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

        if (ballYPos <= 0):
            ballYPos = 0;
            ballYDirection = 1;

        elif (ballYPos >= self.WINDOW_HEIGHT - self.BALL_HEIGHT):
            ballYPos = self.WINDOW_HEIGHT - self.BALL_HEIGHT
            ballYDirection = -1
        return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

    def updatePaddle1(self,action, paddle1YPos):
        if (action[1] == 1):
            paddle1YPos = paddle1YPos - self.PADDLE_SPEED
        if (action[2] == 1):
            paddle1YPos = paddle1YPos + self.PADDLE_SPEED
        if (paddle1YPos < 0):
            paddle1YPos = 0
        if (paddle1YPos > self.WINDOW_HEIGHT - self.PADDLE_HEIGHT):
            paddle1YPos = self.WINDOW_HEIGHT - self.PADDLE_HEIGHT
        return paddle1YPos


    def updatePaddle2(self,paddle2YPos, ballYPos):
        if (paddle2YPos + self.PADDLE_HEIGHT/2 < ballYPos + self.BALL_HEIGHT/2):
            paddle2YPos = paddle2YPos + self.PADDLE_SPEED
        if (paddle2YPos + self.PADDLE_HEIGHT/2 > ballYPos + self.BALL_HEIGHT/2):
            paddle2YPos = paddle2YPos - self.PADDLE_SPEED
        if (paddle2YPos < 0):
            paddle2YPos = 0
        if (paddle2YPos > self.WINDOW_HEIGHT - self.PADDLE_HEIGHT):
            paddle2YPos = self.WINDOW_HEIGHT - self.PADDLE_HEIGHT
        return paddle2YPos
