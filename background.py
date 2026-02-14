
import pygame
import random

pygame.init()
speed_x = -1
speed_y = 1
# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Parallax Bow Background with Random Rotation")

pink_bg = (255, 207, 249)

background = pygame.image.load("assets/background.png").convert_alpha()
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
background.set_alpha(190)  # 0 = fully transparent, 255 = fully opaque

# Pink background

# Load bow image
bow_image = pygame.image.load("assets/bow2.png").convert_alpha()
bow_width, bow_height = bow_image.get_size()

# -----------------------------
# LAYER 1: BIG BOWS
big_bow = pygame.transform.scale(
    bow_image, (bow_width // 10, bow_height // 10))
big_width, big_height = big_bow.get_size()
num_big = 8
big_bows = []
for _ in range(num_big):
    big_bows.append({
        "pos": [random.randint(0, WIDTH), random.randint(0, HEIGHT)],
        "rot": random.uniform(0, 360),                     # random start angle
        # clockwise or counterclockwise
        "rot_speed": random.uniform(0.2, 0.5) * random.choice([-1, 1])
    })

# -----------------------------
# LAYER 2: SMALL BOWS

bow_image.set_alpha(128)  # 50% transparent
small_bow = pygame.transform.scale(
    bow_image, (bow_width // 15, bow_height // 15))
small_width, small_height = small_bow.get_size()
num_small = 12
small_bows = []
for _ in range(num_small):
    small_bows.append({
        "pos": [random.randint(0, WIDTH), random.randint(0, HEIGHT)],
        "rot": random.uniform(0, 360),
        "rot_speed": random.uniform(0.5, 1.2) * random.choice([-1, 1])
    })
num_glitters = 50
glitters = [
    {
        "x": random.randint(0, WIDTH),
        "y": random.randint(0, HEIGHT),
        "size": random.randint(1, 3),
        "speed": random.uniform(1, 3)
    }
    for _ in range(num_glitters)
]

# Hoop placeholder (transparent)
hoop_width, hoop_height = 100, 50
cropped_hoop = pygame.Surface((hoop_width, hoop_height), pygame.SRCALPHA)
hoop_x, hoop_y = 300, 200

# Movement direction

small_speed_scale = 0.5  # small bows move slower

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(pink_bg)
    screen.blit(background, (0, 0))

    for glitter in glitters:
        glitter["y"] += glitter["speed"]
        if glitter["y"] > HEIGHT:
            glitter["y"] = 0
            glitter["x"] = random.randint(0, WIDTH)

    # Flicker effect: change alpha randomly
        alpha = random.randint(100, 255)
        color = (255, 255, 255, alpha)  # white glitter
        s = pygame.Surface((glitter["size"], glitter["size"]), pygame.SRCALPHA)
        pygame.draw.circle(
            s, color, (glitter["size"]//2, glitter["size"]//2), glitter["size"]//2)
        screen.blit(s, (glitter["x"], glitter["y"]))

    # Move and draw big bows with rotation
    for bow in big_bows:
        # Move
        bow["pos"][0] += speed_x
        bow["pos"][1] += speed_y

        # Wrap
        if bow["pos"][1] > HEIGHT:
            bow["pos"][1] = -big_height
            bow["pos"][0] = random.randint(0, WIDTH)
        if bow["pos"][0] > WIDTH:
            bow["pos"][0] = -big_width
        if bow["pos"][0] < -big_width:
            bow["pos"][0] = WIDTH

        # Rotate
        bow["rot"] = (bow["rot"] + bow["rot_speed"]) % 360
        rotated_image = pygame.transform.rotate(big_bow, bow["rot"])
        rect = rotated_image.get_rect(center=bow["pos"])

        # Draw
        screen.blit(rotated_image, rect.topleft)

    # Move and draw small bows with faster rotation
    for bow in small_bows:
        bow["pos"][0] += speed_x * small_speed_scale
        bow["pos"][1] += speed_y * small_speed_scale

        # Wrap
        if bow["pos"][1] > HEIGHT:
            bow["pos"][1] = -small_height
            bow["pos"][0] = random.randint(0, WIDTH)
        if bow["pos"][0] > WIDTH:
            bow["pos"][0] = -small_width
        if bow["pos"][0] < -small_width:
            bow["pos"][0] = WIDTH

        # Rotate
        bow["rot"] = (bow["rot"] + bow["rot_speed"]) % 360
        rotated_image = pygame.transform.rotate(small_bow, bow["rot"])
        rect = rotated_image.get_rect(center=bow["pos"])

        # Draw
        screen.blit(rotated_image, rect.topleft)

    # Draw hoop on top
    screen.blit(cropped_hoop, (hoop_x, hoop_y))

    pygame.display.update()
    clock.tick(60)


pygame.quit()
