import pygame
import random
import os
import neat

pygame.font.init()  # init font

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
STAT_FONT = pygame.font.SysFont("comicsans", 50)
DRAW_LINES = False
SCROLL_SPEED = 5

WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")


pipe_img = pygame.image.load("imgs/pipe.png")
bg_img = pygame.image.load("imgs/bg.png")
bird_images = [pygame.image.load(f"imgs/bird{num}.png") for num in range(1, 4)]
base_img = pygame.transform.scale((pygame.image.load("imgs/ground.png")), (SCREEN_WIDTH + 35, 100))

gen = 0

class Bird():
    GRAVITY = 0.5
    def __init__(self, x, y):
        super().__init__()
        self.images = bird_images
        self.index = 0
        self.counter = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.clicked = False
    
    def jump(self):
        self.vel = -10
        self.clicked = True
    def move(self):
        self.vel += self.GRAVITY
        self.rect.y += self.vel
    def draw(self,win): # Chim dap canh
        self.counter += 1
        flap_cooldown = 5
        if self.counter % flap_cooldown == 0:
            self.index += 1
            if self.index >= len(self.images):
                self.index = 0
        self.image = self.images[self.index]
        win.blit(self.image, self.rect)
        return False
    def update(self):
        self.move()
        self.draw()

class Pipe():
    GAP = 150
    def __init__(self, x):
        super().__init__()
        self.image = pipe_img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.top_pos = 0
        self.bottom_pos = 0

        self.PIPE_TOP = pygame.transform.flip(self.image, False, True)
        self.PIPE_BOTTOM = self.image
        self.passed = False
        self.set_height()
    def set_height(self):
        self.top_pos = random.randint(100, 300) - self.PIPE_TOP.get_height()
        self.bottom_pos = self.top_pos + self.PIPE_TOP.get_height() + self.GAP

    def move(self):
        self.rect.x -= SCROLL_SPEED
    def draw(self,win):
        win.blit(self.PIPE_TOP, (self.rect.x, self.top_pos))
        win.blit(self.PIPE_BOTTOM, (self.rect.x, self.bottom_pos))
    def update(self):
        self.move()
        self.draw(win=WIN)
    def collide(self, bird):
        bird_mask = pygame.mask.from_surface(bird.image)
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.rect.x - bird.rect.x, self.top_pos - round(bird.rect.y))
        bottom_offset = (self.rect.x - bird.rect.x, self.bottom_pos - round(bird.rect.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        if t_point or b_point:
            return True
        return False
    
    
class Base:
    WIDTH = base_img.get_width()
    IMG = base_img
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    def move(self):
        self.x1 -= SCROLL_SPEED
        self.x2 -= SCROLL_SPEED
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    def draw(self,win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
    def update(self):
        self.move()
        self.draw(win=WIN)

def draw_window(win,birds, pipes, base,score,gen):
    win.blit(bg_img, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    # score
    score_label = STAT_FONT.render(f"Score: {score}", 1, (255, 255, 255))
    win.blit(score_label, (10, 10))
    # generations
    score_label = STAT_FONT.render(f"Gen: {gen}", 1, (255, 255, 255))
    win.blit(score_label, (10, 50))
    # alive
    score_label = STAT_FONT.render(f"Alive: {len(birds)}", 1, (255, 255, 255))
    win.blit(score_label, (10, 90))
    
    pygame.display.update()

def generate_pipe(pipes) :
    if len(pipes) == 0:
        pipes.append(Pipe(SCREEN_WIDTH))
    if pipes[-1].rect.x < SCREEN_WIDTH - 250:
        pipes.append(Pipe(SCREEN_WIDTH))
    return pipes



def eval_genomes(genomes, config):
    global WIN,gen
    win = WIN
    gen += 1

    # Create lists to hold the neural network and the genome
    nets = []
    ge = []
    birds = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(200, int(SCREEN_HEIGHT / 2)))
        ge.append(genome)

    base = Base(SCREEN_HEIGHT - base_img.get_height())
    pipes = []
    score = 0

    clock = pygame.time.Clock()
    fps = 60
    run = True
    while run :
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        
        generate_pipe(pipes)

        pipe_index = 0 # index of the pipe the bird is currently looking at
        if len(birds) > 0: # if there are birds alive
            if len(pipes) > 1 and birds[0].rect.x > pipes[0].rect.x + pipes[0].rect.width: # if the bird has passed the first pipe
                pipe_index = 1 # look at the next pipe
        else:
            run = False
            break

        for x, bird in enumerate(birds): # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move()
            inputs = (
                abs(bird.rect.x - pipes[pipe_index].rect.x),
                bird.rect.y,
                abs(bird.rect.y - pipes[pipe_index].top_pos + pipes[pipe_index].PIPE_TOP.get_height()),
                abs(bird.rect.y - pipes[pipe_index].bottom_pos )
            )
            output = nets[birds.index(bird)].activate(inputs)
            if output[0] > 0 and bird.clicked == False:
                bird.jump()
            if output[0] < 0:
                bird.clicked = False
        

        add_score = False
        for pipe in pipes:
            pipe.move()
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    nets.pop(x)
                    ge.pop(x)
                    birds.remove(bird)
            if pipe.rect.x + pipe.rect.width < 0:
                pipes.remove(pipe)
            if not pipe.passed and pipe.rect.x < bird.rect.x:
                pipe.passed = True
                add_score = True

        # add score if the birds passes the pipe
        if add_score: 
            score += 1
            for genome in ge:
                genome.fitness += 5 # give a fitness of 5 for passing a pipe
        
        for x, bird in enumerate(birds):
            if bird.rect.y + bird.rect.height >= SCREEN_HEIGHT - base_img.get_height() or bird.rect.y < 0: # if the bird hits the ground 
                nets.pop(x)
                ge.pop(x)
                birds.remove(bird)
        base.update()
        draw_window(win, birds, pipes, base, score, gen)
        

def run(config_path):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir , "config_feedforward.txt")
    run(config_path)