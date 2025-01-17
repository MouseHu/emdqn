__author__ = 'Batchu Vishal'
import pygame
import math
import numpy as np
from collections import defaultdict
import copy
import sys
import os

from .person import Person
from .onBoard import OnBoard
from .coin import Coin
from .player import Player
from .fireball import Fireball
from .monsterPerson import MonsterPerson


class Board(object):
    '''
    This class defines our gameboard.
    A gameboard contains everthing related to our game on it like our characters, walls, ladders, coins etc
    The generation of the level also happens in this class.
    '''

    def __init__(self, width, height, rewards, rng, _dir, map_config=None, random_noise=None):
        self.__width = width
        self.__actHeight = height
        self.__height = self.__actHeight + 10
        self.score = 0
        self.rng = rng
        self.rewards = rewards
        self.cycles = 0  # For the characters animation
        self.direction = 0
        self._dir = _dir

        self.map_array = None
        self.init_score = 0
        self.init_lives = 1
        self.end_game = defaultdict(bool, {(8, True)})  # end with princess
        self.old_ver = True
        if map_config:
            if 'map_array' in map_config:
                self.map_array = map_config['map_array']
            if 'init_score' in map_config:
                self.init_score = map_config['init_score']
            if 'init_lives' in map_config:
                self.init_lives = map_config['init_lives']
            if 'end_game' in map_config:
                self.end_game = defaultdict(bool, map_config['end_game'])
            if 'dqn_config' in map_config and 'config_name' in map_config['dqn_config'] \
                    and map_config['dqn_config']['config_name'][:2] == 'ez':
                self.old_ver = False

        self.IMAGES = {
            "still": pygame.image.load(os.path.join(_dir, 'assets/still.png')).convert_alpha(),
            "monster0": pygame.image.load(os.path.join(_dir, 'assets/monster0.png')).convert_alpha(),
            "princess": pygame.image.load(os.path.join(_dir, 'assets/princess.png')).convert_alpha(),
            "fireballright": pygame.image.load(os.path.join(_dir, 'assets/fireballright.png')).convert_alpha(),
            "coin1": pygame.image.load(os.path.join(_dir, 'assets/coin1.png')).convert_alpha(),
            "wood_block": pygame.image.load(os.path.join(_dir, 'assets/wood_block.png')).convert_alpha(),
            "ladder": pygame.image.load(os.path.join(_dir, 'assets/ladder.png')).convert_alpha(),
            "spike": pygame.image.load(os.path.join(_dir, 'assets/spike.png')).convert_alpha(),
            "gem5": pygame.image.load(os.path.join(_dir, 'assets/gem5.png')).convert_alpha(),
            "gem6": pygame.image.load(os.path.join(_dir, 'assets/gem6.png')).convert_alpha(),
            "gem7": pygame.image.load(os.path.join(_dir, 'assets/gem7.png')).convert_alpha(),
            "gem8": pygame.image.load(os.path.join(_dir, 'assets/gem8.png')).convert_alpha()
            # "still_bak": pygame.image.load(os.path.join(_dir, 'assets/still_bak.png')).convert_alpha()
        }

        self.white = (255, 255, 255)

        '''
        The map is essentially an array of 30x80 in which we store what each block on our map is.
        1 represents a wall, 2 for a ladder and 3 for a coin.
        '''
        self.map = []
        # These are the arrays in which we store our instances of different
        # classes
        self.Players = []
        self.Enemies = []
        self.Allies = []
        self.Coins = []
        self.Gems = []
        self.Walls = []
        self.Ladders = []
        self.Spikes = []
        self.Fireballs = []
        self.Boards = []
        self.FireballEndpoints = []
        self.CustomObjects = []

        # Resets the above groups and initializes the game for us
        self.resetGroups()

        # Initialize the instance groups which we use to display our instances
        # on the screen
        self.fireballGroup = pygame.sprite.RenderPlain(self.Fireballs)
        self.playerGroup = pygame.sprite.RenderPlain(self.Players)
        self.enemyGroup = pygame.sprite.RenderPlain(self.Enemies)
        self.wallGroup = pygame.sprite.RenderPlain(self.Walls)
        self.ladderGroup = pygame.sprite.RenderPlain(self.Ladders)
        self.spikeGroup = pygame.sprite.RenderPlain(self.Spikes)
        self.coinGroup = pygame.sprite.RenderPlain(self.Coins)
        self.gemGroup = pygame.sprite.RenderPlain(self.Gems)
        self.allyGroup = pygame.sprite.RenderPlain(self.Allies)
        self.fireballEndpointsGroup = pygame.sprite.RenderPlain(
            self.FireballEndpoints)
        self.customObjectsGroup = pygame.sprite.RenderPlain(self.CustomObjects)

        self.random_noise = random_noise
        self.num_frames = 0
        self.noiseimage = None

    def resetGroups(self):
        #  Customized map
        if not self.map_array == None:
            self.score = self.init_score
            self.lives = self.init_lives
            self.map = copy.deepcopy(self.map_array)
            self.Players = []
            self.Allies = []
            self.Coins = []
            self.Gems = []
            self.Walls = []
            self.Ladders = []
            self.Spikes = []
            self.Fireballs = []
            self.FireballEndpoints = []
            self.CustomObjects = []
            self.initializeGameCustomized()
            return

        self.score = 0
        self.lives = 1
        self.map = []  # We will create the map again when we reset the game
        self.Players = [
            Player(
                self.IMAGES["still"],
                (80,
                 141),
                15,
                15)]
        # print("??", self.Players[0].getPosition())
        # self.Enemies = [
        #    MonsterPerson(
        #        self.IMAGES["monster0"],
        #        (240,
        #         100),
        #        self.rng,
        #        self._dir)]
        self.Allies = [Person(self.IMAGES["princess"], (50, 48), 18, 25)]
        self.Allies[0].updateWH(self.Allies[0].image, "H", 0, 25, 25)
        self.Coins = []
        self.Gems = []
        self.Walls = []
        self.Ladders = []
        self.Spikes = []
        self.Fireballs = []
        self.FireballEndpoints = [OnBoard(self.IMAGES["still"], (50, 440))]
        self.CustomObjects = []
        self.initializeGame()  # This initializes the game and generates our map
        self.createGroups()  # This creates the instance groups

    # Checks to destroy a fireball when it reaches its terminal point
    def checkFireballDestroy(self, fireball):
        if pygame.sprite.spritecollide(
                fireball, self.fireballEndpointsGroup, False):
            # We use indices on fireballs to uniquely identify each fireball
            self.DestroyFireball(fireball.index)

    # Creates a new fireball and adds it to our fireball group
    def CreateFireball(self, location, monsterIndex):
        if len(self.Fireballs) < len(self.Enemies) * 5:
            self.Fireballs.append(
                Fireball(self.IMAGES["fireballright"], (location[0], location[1] + 15), len(self.Fireballs),
                         2 + len(self.Enemies) / 2, self.rng, self._dir))
            # Starts monster's animation
            self.Enemies[monsterIndex].setStopDuration(15)
            self.Enemies[monsterIndex].setPosition(
                (self.Enemies[monsterIndex].getPosition()[0], self.Enemies[monsterIndex].getPosition()[1] - 12))
            self.Enemies[monsterIndex].setCenter(
                self.Enemies[monsterIndex].getPosition())
            self.createGroups()  # We recreate the groups so the fireball is added

    # Destroy a fireball if it has collided with a player or reached its
    # endpoint
    def DestroyFireball(self, index):
        for fireBall in range(len(self.Fireballs)):
            if self.Fireballs[fireBall].index == index:
                self.Fireballs.remove(self.Fireballs[fireBall])
                for fireBallrem in range(
                        len(self.Fireballs)):  # We need to reduce the indices of all fireballs greater than this
                    if self.Fireballs[fireBallrem].index > index:
                        self.Fireballs[fireBallrem].index -= 1
                self.createGroups()  # Recreate the groups so the fireball is removed
                break

    # Randomly Generate coins in the level where there is a wall below the
    # coin so the player can reach it
    def GenerateCoins(self):
        for i in range(6, len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] == 0 and ((i + 1 < len(self.map) and self.map[i + 1][j] == 1) or (
                        i + 2 < len(self.map) and self.map[i + 2][j] == 1)):
                    randNumber = math.floor(self.rng.rand() * 1000)
                    if randNumber % 35 == 0 and len(
                            self.Coins) <= 25:  # At max there will be 26 coins in the map
                        self.map[i][j] = 3
                        if j - 1 >= 0 and self.map[i][j - 1] == 3:
                            self.map[i][j] = 0
                        if self.map[i][j] == 3:
                            # Add the coin to our coin list
                            self.Coins.append(
                                Coin(
                                    self.IMAGES["coin1"],
                                    (j * 15 + 15 / 2,
                                     i * 15 + 15 / 2),
                                    self._dir))
        if len(self.Coins) <= 15:  # If there are less than 21 coins, we call the function again
            self.GenerateCoins()

    # Given a position and checkNo ( 1 for wall, 2 for ladder, 3 for coin) the
    # function tells us if its a valid position to place or not
    def checkMapForMatch(self, placePosition, floor, checkNo, offset):
        if floor < 1:
            return 0
        for i in range(
                0, 2):  # We will get things placed atleast 2-1 blocks away from each other
            if self.map[floor * 5 - offset][placePosition + i] == checkNo:
                return 1
            if self.map[floor * 5 - offset][placePosition - i] == checkNo:
                return 1
        return 0

    # Create an empty 2D map of 30x30 size (32x32 including boundaries)
    def makeMap(self):
        for point in range(0, int(self.__height / 15 + 1)):
            row = []
            for point2 in range(0, int(self.__width / 15)):
                row.append(0)
            self.map.append(row)

    # Add walls to our map boundaries and also the floors
    def makeWalls(self):
        for i in range(0, int(self.__height / 15)):
            self.map[i][0] = self.map[i][int(self.__width / 15 - 1)] = 1
        for i in range(2, int(self.__height / (15 * 5) + 1)):
            for j in range(0, int(self.__width / 15)):
                self.map[i * 5][j] = 1
                if i == 2 and (j > 10):
                    self.map[i * 5 - 1][j] = 4

    # Make a small chamber on the top where the princess resides
    def makePrincessChamber(self):
        for j in range(0, 4):
            self.map[j][9] = 1

        for j in range(0, 10):
            self.map[4][j] = 1

        for j in range(0, 6):
            self.map[1 * 4 + j][7] = self.map[1 * 4 + j][8] = 2
            # self.map[1 * 4 + j][5] = self.map[1 * 4 + j][6] = \
            # self.map[1 * 4 + j][7] = self.map[1 * 4 + j][8] = 2

    # Generate ladders randomly, 1 for each floor such that they are not too
    # close to each other
    def makeLadders(self):
        pass
        ## [y][x]
        # for i in range(2, int(self.__height / (15 * 4) - 1)):
        #    ladderPos = math.floor(self.rng.rand() * (self.__width / 15 - 20))
        #    ladderPos = int(7 + ladderPos)
        #    while self.checkMapForMatch(ladderPos, i - 1, 2, 0) == 1:#
        #        ladderPos = math.floor(
        #            self.rng.rand() * (self.__width / 15 - 20))
        #        ladderPos = int(7 + ladderPos)
        #    for k in range(0, 5):
        #        self.map[i * 5 + k][ladderPos] = s#elf.map[i * 5 + k][ladderPos + 1] = 2
        #        #        self.map[i * 5 + k][ladderPos + 2] = self.map[i * 5 + k][ladderPos - 1] = 2

    # Create the holes on each floor (extreme right and extreme left)
    def makeHoles(self):
        for i in range(3, int(self.__height / (15 * 5))):
            for k in range(
                    1, 3):  # Ladders wont interfere since they leave 10 blocks on either side
                if i % 2 == 0:
                    self.map[i * 5][k] = 0
                else:
                    self.map[i * 5][int(self.__width / 15 - 1 - k)] = 0

    '''
    This is called once you have finished making holes, ladders, walls etc
    You use the 2D map to add instances to the groups
    '''

    def populateMap(self):
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                if self.map[x][y] == 1:
                    # Add a wall at that position
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["wood_block"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))
                elif self.map[x][y] == 2:
                    # Add a ladder at that position
                    self.Ladders.append(
                        OnBoard(
                            self.IMAGES["ladder"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))
                elif self.map[x][y] == 4:
                    # Add a spike at that position
                    self.Spikes.append(
                        OnBoard(
                            self.IMAGES["spike"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))

    # Map customization
    def singletonMapDict(self, key=None):
        if 'map_dict' not in self.__dict__:
            def get_list(key):
                return self.Allies if self.end_game[key] else self.CustomObjects

            self.map_dict = {
                ### 0: ('image_name of nothing', list_to_append, type_to_append, (offset_y, offset_x), additional_args)
                1: ('wood_block', self.Walls, OnBoard, (15 / 2, 15 / 2), []),
                2: ('ladder', self.Ladders, OnBoard, (15 / 2, 15 / 2), []),
                3: ('coin1', self.Coins, Coin, (15 / 2, 15 / 2), [self._dir]),
                4: ('spike', self.Spikes, OnBoard, (15 / 2, 15 / 2), []),
                5: ('gem8', self.Gems, Person, (15 / 2, 15 / 2), [13, 13]),
                6: ('gem6', get_list(6), Person, (15 / 2, 15 / 2), [14, 14]),
                7: ('coin1', get_list(7), Person, (15 / 2, 15 / 2), [12, 12]),
                8: ('princess', get_list(8), Person, (15 / 2, - 25 / 2 + 15), [18, 25]),
                9: ('still', self.Players, Player, (15 / 2, 15 / 2), [15, 15]),  # the player
                10: ('coin1', self.Fireballs, Fireball, (15 / 2, 15 / 2), [0, 3, 0, self._dir]),
            } if self.old_ver else {
                ### 0: ('image_name of nothing', list_to_append, type_to_append, (offset_y, offset_x), additional_args)
                1: ('wood_block', self.Walls, OnBoard, (15 / 2, 15 / 2), []),
                2: ('ladder', self.Ladders, OnBoard, (15 / 2, 15 / 2), []),
                3: ('coin1', self.Coins, Coin, (15 / 2, 15 / 2), [self._dir]),
                4: ('spike', self.Spikes, OnBoard, (15 / 2, 15 / 2), []),
                5: ('gem8', self.Gems, Person, (15 / 2, 15 / 2), [14, 14]),
                6: ('gem6', get_list(6), Person, (15 / 2, 15 / 2), [14, 14]),
                7: ('coin1', get_list(7), Person, (15 / 2, 15 / 2), [14, 14]),
                8: ('princess', get_list(8), Person, (15 / 2, - 25 / 2 + 15), [18, 25]),
                9: ('still', self.Players, Player, (15 / 2, 15 / 2), [15, 15]),  # the player
                # 10: ('coin1', get_list(7), OnBoard, (15 / 2, 15 / 2), None),
            }
        return self.map_dict[key] if key else self.map_dict

    def populateMapCustomized(self):
        playerPos = []
        for x in range(len(self.map)):  # x and y are inverted
            for y in range(len(self.map[x])):
                if self.map[x][y] == 9:
                    # Add the position to playerPos
                    playerPos.append((x, y))
                    continue

                map_dict = self.singletonMapDict()
                if not self.map[x][y] in map_dict:
                    # Nothing to be placed
                    continue
                self.append_list(x, y, map_dict[self.map[x][y]])

        # place the player randomly
        # huhao finds here
        index = np.random.randint(len(playerPos))
        self.player_pos = playerPos[index]
        self.reset_player_pos()

    def reset_player_pos(self):
        x, y = self.player_pos
        self.append_list(x, y, self.singletonMapDict(9))

    def append_list(self, x, y, info):
        info[1].append(  # list_to_append
            info[2](  # type_to_append
                self.IMAGES[info[0]],
                (y * 15 + info[3][0],
                 x * 15 + info[3][1]),
                *info[4]
            ))

    # Check if the player is on a ladder or not
    def ladderCheck(self, laddersCollidedBelow,
                    wallsCollidedBelow, wallsCollidedAbove):
        if laddersCollidedBelow and len(wallsCollidedBelow) == 0:
            # modified by huhao add check for exact collided
            for ladder in laddersCollidedBelow:
                if ladder.getPosition()[1] + 2 >= self.Players[0].getPosition()[1]:
                    self.Players[0].onLadder = 1
                    self.Players[0].isJumping = 0
                    # Move the player down if he collides a wall above
                    if wallsCollidedAbove:
                        self.Players[0].updateY(3)
        else:
            self.Players[0].onLadder = 0

    # Update all the fireball positions and check for collisions with player
    def fireballCheck(self):
        for fireball in self.fireballGroup:
            fireball.continuousUpdate(self.wallGroup, self.ladderGroup)
            if fireball.checkCollision(self.playerGroup, "V"):
                self.Fireballs.remove(fireball)
                self.Players[0].setPosition((50, 435))
                self.score += self.rewards["negative"]
                self.lives += -1
                self.createGroups()
            self.checkFireballDestroy(fireball)

    # Check for spike collided and add the appropriate score
    def spikeCheck(self):
        spikeCollided = pygame.sprite.spritecollide(
            self.Players[0], self.spikeGroup, False)
        if len(spikeCollided) > 0:
            self.Players[0].setPosition((100, 100))
            self.score += self.rewards["negative"]
            self.lives += -1
            self.createGroups()

    # Check for coins collided and add the appropriate score
    def gemCheck(self, gemCollected):
        for gem in gemCollected:
            self.score += self.rewards["positive"]
            self.createGroups()

    def coinCheck(self, coinsCollected):
        for coin in coinsCollected:
            self.score += self.rewards["positive"]
            # We also remove the coin entry from our map
            self.map[int((coin.getPosition()[1] - 15 / 2) /
                         15)][int((coin.getPosition()[0] - 15 / 2) / 15)] = 0
            # Remove the coin entry from our list
            self.Coins.remove(coin)
            # Update the coin group since we modified the coin list
            self.createGroups()

    # Check if the player wins
    def checkVictory(self):
        # If you touch the princess or reach the floor with the princess you
        # win!
        if self.Players[0].checkCollision(self.allyGroup):
            # or self.Players[0].getPosition()[1] < 4 * 15:
            # if self.Players[0].getPosition()[0] == 50 and self.Players[0].getPosition()[1] == 48:

            self.score += self.rewards["win"]
            self.lives += -1
            # This is just the next level so we only clear the fireballs and
            # regenerate the coins
            self.Fireballs = []
            self.Players[0].setPosition((50, 440))
            self.Coins = []
            # self.GenerateCoins()

            # Add monsters
            # if len(self.Enemies) == 1:
            #     self.Enemies.append(
            #         MonsterPerson(
            #             self.IMAGES["monster0"], (700, 117), self.rng, self._dir))
            # elif len(self.Enemies) == 2:
            #     self.Enemies.append(
            #         MonsterPerson(
            #             self.IMAGES["monster0"], (400, 117), self.rng, self._dir))
            # Create the groups again so the enemies are effected
            self.createGroups()

    def add_noise(self, screen):

        background_color = np.array([120, 60, 120])
        if self.random_noise is None:
            return

        if self.num_frames % 21 == 0 or self.noiseimage is None:
            # self.noiseimage = pygame.Surface()
            random_noise_selected = self.random_noise[np.random.randint(0, len(self.random_noise))]
            self.noiseimage = pygame.surfarray.make_surface(background_color + random_noise_selected)
        screen.blit(self.noiseimage, (0, 0))

    # Redraws the entire game screen for us
    def redrawScreen(self, screen, width, height, noisy=False):
        self.num_frames += 1
        if noisy:
            screen.fill([120, 60, 120])  # Fill it with black
            self.add_noise(screen)
        else:
            screen.fill([120, 60, 120])  # Fill it with black
        # Draw all our groups on the background
        self.ladderGroup.draw(screen)
        self.spikeGroup.draw(screen)
        self.coinGroup.draw(screen)
        self.gemGroup.draw(screen)
        self.wallGroup.draw(screen)
        self.fireballGroup.draw(screen)
        self.enemyGroup.draw(screen)
        self.allyGroup.draw(screen)
        self.customObjectsGroup.draw(screen)
        self.playerGroup.draw(screen)

    # Update all the groups from their corresponding lists
    def createGroups(self):
        self.fireballGroup = pygame.sprite.RenderPlain(self.Fireballs)
        self.playerGroup = pygame.sprite.RenderPlain(self.Players)
        self.enemyGroup = pygame.sprite.RenderPlain(self.Enemies)
        self.wallGroup = pygame.sprite.RenderPlain(self.Walls)
        self.ladderGroup = pygame.sprite.RenderPlain(self.Ladders)
        self.spikeGroup = pygame.sprite.RenderPlain(self.Spikes)
        self.coinGroup = pygame.sprite.RenderPlain(self.Coins)
        self.gemGroup = pygame.sprite.RenderPlain(self.Gems)
        self.allyGroup = pygame.sprite.RenderPlain(self.Allies)
        self.fireballEndpointsGroup = pygame.sprite.RenderPlain(
            self.FireballEndpoints)
        self.customObjectsGroup = pygame.sprite.RenderPlain(self.CustomObjects)

    '''
    Initialize the game by making the map, generating walls, generating princess chamber, generating ladders randomly,
    generating broken ladders randomly, generating holes, generating coins randomly, adding the ladders and walls to our lists
    and finally updating the groups.
    '''

    def initializeGame(self):
        self.makeMap()
        self.makeWalls()
        self.makePrincessChamber()
        self.makeLadders()
        self.makeHoles()
        # self.GenerateCoins()
        self.populateMap()
        self.createGroups()

    def initializeGameCustomized(self):
        self.populateMapCustomized()
