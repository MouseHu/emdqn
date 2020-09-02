__author__ = 'Batchu Vishal'
import pygame

'''
This class defines all living things in the game, ex.Donkey Kong, Player etc
Each of these objects can move in any direction specified.
'''


class NoisyBackground(pygame.sprite.Sprite):

    def __init__(self, raw_image, position, width, height):
        super(NoisyBackground, self).__init__()
        self.width = width
        self.height = height
        self.__position = position
        self.image = raw_image
        self.image = pygame.transform.scale(
            self.image, (width, height)).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = self.__position