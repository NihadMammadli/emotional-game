import pygame
import sys
import random

pygame.init()

WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 20
OBSTACLE_WIDTH = 50
OBSTACLE_SPEED = 10
FPS = 60
OBSTACLE_HEIGHT = 300  

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Endless Obstacle Game")

font = pygame.font.Font(None, 36)

ball_x = WIDTH // 4
ball_y = HEIGHT // 2

obstacles = [{"x": WIDTH, "gap_top": random.randint(50, HEIGHT - OBSTACLE_HEIGHT - 50), "obstacle_height": random.randint(200, 400)}]

clock = pygame.time.Clock()

obstacle_speed = OBSTACLE_SPEED
ball_speed = 5
passed_obstacles = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        ball_y -= ball_speed
    elif keys[pygame.K_DOWN]:
        ball_y += ball_speed

    for obstacle in obstacles:
        obstacle["x"] -= obstacle_speed

    if obstacles[-1]["x"] < WIDTH - WIDTH // 3:
        new_obstacle = {"x": WIDTH, "gap_top": random.randint(50, HEIGHT - OBSTACLE_HEIGHT - 50), "obstacle_height": random.randint(150, 300)}
        obstacles.append(new_obstacle)

    obstacles = [obstacle for obstacle in obstacles if obstacle["x"] + OBSTACLE_WIDTH > 0]

    for obstacle in obstacles:
        if (
            ball_x + BALL_RADIUS > obstacle["x"]
            and ball_x - BALL_RADIUS < obstacle["x"] + OBSTACLE_WIDTH
            and not (obstacle["gap_top"] < ball_y < obstacle["gap_top"] + obstacle["obstacle_height"])
        ):
            print("Game Over!")
            pygame.quit()
            sys.exit()

    for obstacle in obstacles:
        if obstacle["x"] + OBSTACLE_WIDTH < ball_x - BALL_RADIUS:
            passed_obstacles += 1
            if passed_obstacles % 3 == 0:
                obstacle_speed += 0.1  
                ball_speed += 0.1 

    screen.fill(WHITE)

    pygame.draw.circle(screen, BLACK, (ball_x, ball_y), BALL_RADIUS)

    for obstacle in obstacles:
        gap_top = obstacle["gap_top"]
        gap_bottom = HEIGHT - gap_top - obstacle["obstacle_height"]

        pygame.draw.rect(screen, BLACK, (obstacle["x"], 0, OBSTACLE_WIDTH, gap_top))
        pygame.draw.rect(screen, BLACK, (obstacle["x"], gap_top + obstacle["obstacle_height"], OBSTACLE_WIDTH, gap_bottom))

    pygame.display.flip()

    clock.tick(FPS)
