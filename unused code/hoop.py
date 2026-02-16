
import pygame

pygame.init()
# initial positions
hoop_x = 50
hoop_y = 100

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cropped & Recolored Hoop with Tolerance")

# 2️⃣ Define color distance function


def color_close(c1, c2, tolerance=30):
    """
    Returns True if RGB colors c1 and c2 are within the tolerance
    """
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5 < tolerance


# 3️⃣ Load image
hoop_image = pygame.image.load("assets/hoopnobgd.png").convert_alpha()
width, height = hoop_image.get_size()

# 4️⃣ Crop left half
# Load and crop hoop as before


crop_rect = pygame.Rect(width // 2, 0, width // 2, height)  # keep right half

# 4️⃣ Create a new surface and blit the cropped part
cropped_hoop = pygame.Surface((width//2, height), pygame.SRCALPHA)
cropped_hoop.blit(hoop_image, (0, 0), crop_rect)

# Fill inside the hoop rim with pink
start_inside_hoop = (cropped_hoop.get_width()//2, cropped_hoop.get_height()//2)

# 5️⃣ Color replacement mapping
color_map = {
    (198, 27, 7): (255, 46, 214),    # red rgb(255, 46, 214)
    (238, 63, 44): (255, 105, 226),   # light red rgb(255, 105, 226)
    (170, 187, 87): (125, 79, 116),  # dark grey
    (217, 226, 235): (255, 105, 226),  # light grey
    (255, 126, 7): (255, 125, 243),  # orange
    (221, 94, 0): (201, 97, 192),    # dark orange
    (255, 255, 255): (255, 125, 243),  # white → pink
}

tolerance = 60  # how much variation is allowed in the color replacement

# 6️⃣ Replace colors with tolerance
cropped_hoop.lock()
for x in range(cropped_hoop.get_width()):
    for y in range(cropped_hoop.get_height()):
        pixel = cropped_hoop.get_at((x, y))
        rgb = pixel[:3]
        for old_rgb, new_rgb in color_map.items():
            if color_close(rgb, old_rgb, tolerance):
                cropped_hoop.set_at((x, y), (*new_rgb, pixel[3]))
                break
cropped_hoop.unlock()


# 7️⃣ Display loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))
    screen.blit(cropped_hoop, (hoop_x, hoop_y))
    pygame.display.update()
    clock.tick(60)

pygame.quit()
