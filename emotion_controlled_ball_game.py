import pygame
import sys
import random
import math
import cv2
from keras.models import load_model
import numpy as np

pygame.init()

WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 20
OBSTACLE_WIDTH = 50
OBSTACLE_SPEED = 10
FPS = 60
OBSTACLE_HEIGHT = 300

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

ball_image = pygame.image.load("assets/images/neonball.png")
ball_image = pygame.transform.scale(ball_image, (BALL_RADIUS * 2, BALL_RADIUS * 2))

background_image = pygame.image.load("assets/images/background.jpg")
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Emotional Bird Game")

font = pygame.font.Font(None, 36)

ball_x = WIDTH // 4
ball_y = HEIGHT // 2

obstacles = [{"x": WIDTH, "gap_top": random.randint(50, HEIGHT - OBSTACLE_HEIGHT - 50), "obstacle_height": random.randint(200, 300)}]

clock = pygame.time.Clock()

obstacle_speed = OBSTACLE_SPEED
ball_speed = 5
passed_obstacles = 0
points = 0

emotion_model = load_model('assets/models/model_v6_23.hdf5', compile=False)
emotions = ['Sad', 'Neutral', 'Happy', 'Sad', 'Happy', 'Sad', 'Neutral']
threshold = 0.2

vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('assets/models/haarcascade_frontalface.xml')

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

    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 9)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        emotion_prediction = emotion_model.predict(roi_gray)
        max_index = np.argmax(emotion_prediction)
        max_emotion = emotions[max_index]

        if max_emotion == "Happy":
            ball_y += ball_speed
        elif max_emotion == "Neutral":
            ball_y -= ball_speed

        cv2.putText(frame, f'Emotion: {max_emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        


    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(1) 
    keys = pygame.key.get_pressed()

    for obstacle in obstacles:
        obstacle["x"] -= obstacle_speed

    if obstacles[-1]["x"] < WIDTH - WIDTH // 3:
        new_obstacle = {"x": WIDTH, "gap_top": random.randint(50, HEIGHT - OBSTACLE_HEIGHT - 50), "obstacle_height": random.randint(200, 300)}
        obstacles.append(new_obstacle)

    obstacles = [obstacle for obstacle in obstacles if obstacle["x"] + OBSTACLE_WIDTH > 0]

    for obstacle in obstacles:
        if (
            ball_x + BALL_RADIUS > obstacle["x"]
            and ball_x - BALL_RADIUS < obstacle["x"] + OBSTACLE_WIDTH
            and not (obstacle["gap_top"] < ball_y < obstacle["gap_top"] + obstacle["obstacle_height"])
        ):
            print("Game Over! Points:", points)
            pygame.quit()
            sys.exit()

    for obstacle in obstacles:
        if obstacle["x"] + OBSTACLE_WIDTH < ball_x - BALL_RADIUS:
            passed_obstacles += 1
            if passed_obstacles % 3 == 0:
                obstacle_speed += 0.05
                ball_speed += 0.05
            points += 0.1

    screen.blit(background_image, (0, 0))

    screen.blit(ball_image, (ball_x - BALL_RADIUS, ball_y - BALL_RADIUS))

    for obstacle in obstacles:
        gap_top = obstacle["gap_top"]
        gap_bottom = HEIGHT - gap_top - obstacle["obstacle_height"]

        pygame.draw.rect(screen, BLACK, (obstacle["x"], 0, OBSTACLE_WIDTH, gap_top))
        pygame.draw.rect(screen, BLACK, (obstacle["x"], gap_top + obstacle["obstacle_height"], OBSTACLE_WIDTH, gap_bottom))

    points_text = font.render(f"Points: {math.ceil(points)}", True, RED)
    screen.blit(points_text, (10, 10))

    pygame.display.flip()

    clock.tick(FPS)
