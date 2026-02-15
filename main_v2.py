from email.mime import image
import math
from operator import pos
import pygame
import matplotlib as plt

# ----------------- config -----------------
W, H = 1000, 650
FPS = 120
DT = 1.0 / 240.0 

WHITE = (245, 245, 245)
BLACK = (25, 25, 25)
RED   = (220, 60, 60)
BLUE  = (50, 90, 220)
GRAY  = (180, 180, 180)
GREEN = (60, 190, 90)

SKY_TOP = (235, 245, 255)
SKY_BOT = (210, 230, 255)
COURT   = (245, 230, 210)
COURT_LINE = (210, 190, 170)

PX_PER_M = 220

def m2px(v): 
    return v * PX_PER_M

def world_to_screen(pos_m):
    return (int(m2px(pos_m[0])), int(m2px(pos_m[1])))

def lerp(a, b, t): 
    return a + (b - a) * t

def draw_vertical_gradient(screen, top_color, bottom_color):
    # simple gradient: draw horizontal lines
    for y in range(H):
        t = y / (H - 1)
        c = (
            int(lerp(top_color[0], bottom_color[0], t)),
            int(lerp(top_color[1], bottom_color[1], t)),
            int(lerp(top_color[2], bottom_color[2], t)),
        )
        pygame.draw.line(screen, c, (0, y), (W, y))

def aa_circle(screen, color, center, radius):
    # pygame's built-in draw is okay; gfxdraw is nicer if available
    try:
        import pygame.gfxdraw as gfx
        gfx.filled_circle(screen, center[0], center[1], radius, color)
        gfx.aacircle(screen, center[0], center[1], radius, color)
    except Exception:
        pygame.draw.circle(screen, color, center, radius)

def aa_line(screen, color, p1, p2, width=2):
    # aaline is thin; for thicker lines draw multiple
    if width <= 1:
        pygame.draw.aaline(screen, color, p1, p2)
    else:
        pygame.draw.line(screen, color, p1, p2, width)

def draw_text(screen, font, text, x, y, color=BLACK):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

#images global variables
im_green = False
im_moonshot = False
flash_start_time = 0
flash_duration = 1000  # milliseconds



# ----------------- physics objects -----------------
class Pendulum:
    """Small-angle analytic pendulum: theta = A cos(omega t + phi)."""
    def __init__(self, pivot_m=(1.5, 1.2), L=1.2, A=0.75, g=9.81, phi=0.0):
        self.pivot = pivot_m
        self.L = L
        self.A = A
        self.g = g
        self.phi = phi
        self.t = 0.0

    def omega(self):
        return math.sqrt(max(self.g, 0.0) / max(self.L, 1e-6))

    def theta(self):
        return self.A * math.cos(self.omega() * self.t + self.phi)

    def theta_dot(self):
        return -self.A * self.omega() * math.sin(self.omega() * self.t + self.phi)

    def ball_pos(self):
        th = self.theta()
        x = self.pivot[0] + self.L * math.sin(th)
        y = self.pivot[1] + self.L * math.cos(th)   # FIX: y-down world
        return (x, y)

    def ball_vel(self):
        th = self.theta()
        thd = self.theta_dot()
        vx = self.L * thd * math.cos(th)
        vy = -self.L * thd * math.sin(th)           # FIX sign to match y expression
        return (vx, vy)

    def step(self, dt):
        self.t += dt

class Ball:
    def __init__(self, pos_m=(0, 0), vel_mps=(0, 0), radius_m=0.12):
        self.pos = pos_m
        self.vel = vel_mps
        self.r = radius_m
        self.attached = True

    def attach_to(self, pendulum: Pendulum):
        self.attached = True
        self.pos = pendulum.ball_pos()
        self.vel = (0.0, 0.0)

    def release_from(self, pendulum: Pendulum):
        self.attached = False
        self.pos = pendulum.ball_pos()
        self.vel = pendulum.ball_vel()

    def step(self, dt, g=9.81, wind_ax=0.0):
        if self.attached:
            return
        ay = g  # y-down gravity is +g
        self.vel = (self.vel[0], self.vel[1] + ay * dt)
        self.pos = (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt)

class Hoop:
    """Score zone as a circle, plus a nice drawing helper."""
    def __init__(self, center_m=(2.0, 2.1), radius_m=0.18):
        self.c = center_m
        self.r = radius_m

    def scored(self, ball: Ball):
        dx = ball.pos[0] - self.c[0]
        dy = ball.pos[1] - self.c[1]
        return (dx*dx + dy*dy) <= (self.r * self.r)

    def draw(self, screen):
        cpx = world_to_screen(self.c)
        rim_r = int(m2px(self.r))

        # backboard
        board_w, board_h = 10, 120
        board_x = cpx[0] + rim_r + 5
        board_y = cpx[1] - board_h // 2 - 30
        pygame.draw.rect(screen, (250, 250, 250), (board_x, board_y, board_w, board_h), border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), (board_x, board_y, board_w, board_h), 3, border_radius=6)

        # rim
        aa_circle(screen, RED, cpx, rim_r)
        aa_circle(screen, WHITE, cpx, max(1, rim_r - 6))

        # net (simple)
        for i in range(-3, 4):
            x0 = cpx[0] + i * (rim_r // 3)
            pygame.draw.line(screen, (160, 160, 160), (x0, cpx[1] + 5), (x0 - i * 3, cpx[1] + rim_r + 18), 2)

# ----------------- main loop -----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pendulum Basketball (Upgraded Look)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    bigfont = pygame.font.SysFont("consolas", 28, bold=True)

    g = 9.81
    L = 1.35
    A = 0.9

    pend = Pendulum(pivot_m=(1.2, -1.0), L=L, A=A, g=g, phi=0.3)
    ball = Ball(radius_m=0.12)
    ball.attach_to(pend)
    hoop = Hoop(center_m=(3.8, 2.15), radius_m=0.20)

    #images for score / miss
    green_fn = pygame.image.load("green_fn.png").convert_alpha()
    curry_moonshot = pygame.image.load("curry_moonshot.png").convert_alpha()
    global im_green, im_moonshot, flash_start_time, flash_duration

    score = 0
    level = 1

    trail = []  # list of recent pixel positions
    TRAIL_MAX = 28

    accumulator = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    pend.t = 0.0
                    ball.attach_to(pend)
                    trail.clear()
                if event.key == pygame.K_SPACE:
                    if ball.attached:
                        ball.release_from(pend)
                        trail.clear()

        frame_dt = clock.tick(FPS) / 1000.0
        accumulator += frame_dt

    
        while accumulator >= DT:
            pend.step(DT)

            if ball.attached:
                ball.pos = pend.ball_pos()
            else:
                ball.step(DT, g=g)

                # score check
                if hoop.scored(ball):
                    score += 1
                    level += 1
                    #flash_image(screen, green_fn, (0,0), 5.0)
                    im_green = True
                    flash_start_time = pygame.time.get_ticks()


                    pend.t = 0.0
                    ball.attach_to(pend)
                    trail.clear()
                
                else: 
                    if ball.pos[1]>2.5 or ball.pos[0]<0 or ball.pos[0]>6:
                        im_moonshot = True
                        flash_start_time = pygame.time.get_ticks()

                # out of bounds reset
                if ball.pos[0] < -1 or ball.pos[0] > 6 or ball.pos[1] > 4:
                    pend.t = 0.0
                    ball.attach_to(pend)
                    trail.clear()

            accumulator -= DT

        # ----- DRAW -----
        draw_vertical_gradient(screen, SKY_TOP, SKY_BOT)
        
        # court strip at bottom
        court_y = int(H * 0.78)
        pygame.draw.rect(screen, COURT, (0, court_y, W, H - court_y))
        pygame.draw.line(screen, COURT_LINE, (0, court_y), (W, court_y), 4)

        # draw hoop
        hoop.draw(screen)

        # draw pivot + rod
        pivot_px = world_to_screen(pend.pivot)
        ball_px = world_to_screen(ball.pos)
        aa_circle(screen, BLACK, pivot_px, 6)
        aa_line(screen, (120, 120, 120), pivot_px, ball_px, width=5)

        # trail after release
        if not ball.attached:
            trail.append(ball_px)
            if len(trail) > TRAIL_MAX:
                trail.pop(0)
            for i in range(1, len(trail)):
                # fade by drawing lighter lines for older segments
                pygame.draw.line(screen, (140, 160, 200), trail[i-1], trail[i], 3)

        # ball shadow (fake depth)
        # more shadow when lower on screen
        shadow_strength = max(0.15, min(0.65, (ball_px[1] / H)))
        shadow_w = int(m2px(ball.r) * (1.6 + 0.8 * shadow_strength))
        shadow_h = int(m2px(ball.r) * (0.55 + 0.3 * shadow_strength))
        shadow_surf = pygame.Surface((shadow_w*2, shadow_h*2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, int(120 * shadow_strength)), shadow_surf.get_rect())
        screen.blit(shadow_surf, (ball_px[0] - shadow_w, court_y - shadow_h))

        # draw ball
        aa_circle(screen, BLUE, ball_px, int(m2px(ball.r)))
        aa_circle(screen, (240, 240, 255), (ball_px[0] - 6, ball_px[1] - 6), max(2, int(m2px(ball.r) * 0.35)))

        # HUD
        draw_text(screen, font, f"L={pend.L:.2f} m   A={pend.A:.2f} rad   g={g:.2f}", 20, 18)
        draw_text(screen, font, "SPACE=release   R=reset", 20, 62)
        draw_text(screen, font, f"x position ball: {ball.pos[0]:.2f} m", 20, 84)
        draw_text(screen, font, f"y position ball: {ball.pos[1]:.2f} m", 20, 106)  

        title = bigfont.render(f"Score: {score}   Level: {level}", True, (30, 30, 30))
        screen.blit(title, (W - title.get_width() - 20, 16))

        if im_green:
            elapsed = pygame.time.get_ticks() - flash_start_time
            alpha = max(255 - 255 * (elapsed / flash_duration), 0)

            image_copy = green_fn.copy()
            image_copy.set_alpha(int(alpha))
            screen.blit(image_copy, (0, 0))

            if alpha <= 0:
                im_green = False  # flash finished

        if im_moonshot: #and ball.pos[1]>2.5:
            elapsed = pygame.time.get_ticks() - flash_start_time
            alpha = max(255 - 255 * (elapsed / flash_duration), 0)

            image_copy = curry_moonshot.copy()
            image_copy.set_alpha(int(alpha))
            screen.blit(image_copy, (0, 0))

            if alpha <= 0:
                im_moonshot = False  # flash finished
        
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()