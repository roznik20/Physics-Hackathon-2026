import pygame

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rotating Ball")

# Clock to control framerate
clock = pygame.time.Clock()

# Load ball image (make sure this exists in assets/)
ball_image = pygame.image.load("assets/ball.png").convert_alpha()
ball_width, ball_height = ball_image.get_size()
ball_image = pygame.transform.scale(ball_image, (ball_width // 10, ball_height // 10))

# Ball properties
ball_x, ball_y = 400, 300       # initial position
ball_speed_x, ball_speed_y = 3, 0  # velocity
ball_angle = 0                  # starting rotation angle
ball_rotation_speed = 5         # degrees per frame

running = True
while running:
    screen.fill((255, 207, 249))  # pink background

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move ball
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Update rotation
    ball_angle += ball_rotation_speed

    # Rotate the ball
    rotated_ball = pygame.transform.rotate(ball_image, ball_angle)
    rotated_rect = rotated_ball.get_rect(center=(ball_x, ball_y))

    # Draw background

    # Draw ball
    screen.blit(rotated_ball, rotated_rect)

    # Update the display
    pygame.display.update()
    clock.tick(60)  # 60 FPS

pygame.quit()
