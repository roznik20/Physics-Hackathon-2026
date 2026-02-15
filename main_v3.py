from email.mime import image
import math
from operator import pos
import pygame
import background as bg

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
        ax, ay = wind_ax, g  # y-down gravity is +g
        self.vel = (self.vel[0] + ax * dt, self.vel[1] + ay * dt)
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
        my_game = Game()
        cpx = my_game.world_to_screen(self.c)
        rim_r = int(my_game.m2px(self.r))

        # backboard
        board_w, board_h = 10, 120
        board_x = cpx[0] + rim_r + 5
        board_y = cpx[1] - board_h // 2 - 30
        pygame.draw.rect(screen, (250, 250, 250), (board_x, board_y, board_w, board_h), border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), (board_x, board_y, board_w, board_h), 3, border_radius=6)

        # rim
        my_game.aa_circle(screen, my_game.red, cpx, rim_r)
        my_game.aa_circle(screen, my_game.white, cpx, max(1, rim_r - 6))

        # net (simple)
        for i in range(-3, 4):
            x0 = cpx[0] + i * (rim_r // 3)
            pygame.draw.line(screen, (160, 160, 160), (x0, cpx[1] + 5), (x0 - i * 3, cpx[1] + rim_r + 18), 2)


#----------------- States creation -----------------

class Menu:

    def __init__(self):
        
        self.screen_width = 1000
        self.screen_height = 650

        # Button setup
        self.button_width = 200
        self.button_height = 80

        self.button_rect = pygame.Rect(
            self.screen_width // 2 - self.button_width // 2,
            self.screen_height // 2 - self.button_height // 2,
            self.button_width,
            self.button_height
        )

        self.font = pygame.font.SysFont(None, 48)
        self.title_font = pygame.font.SysFont(None, 100)
        self.title_color = (10,10,10)

        self.bg_color = (30, 30, 30)
        self.button_color = (255, 153, 204)
        self.button_hover_color = (218, 112, 214)
        self.text_color = (255, 255, 255)

        #bg images
        self.bliss = pygame.image.load("assets/bliss.jpg")
        self.lebron = pygame.image.load("assets/lebron.png")

        self.bliss_scaled, self.bliss_pos = self.scale_and_center_image(self.bliss)
        self.lebron_scaled, self.lebron_pos = self.scale_and_center_image(self.lebron)

        # render the text
        self.title_surf = self.title_font.render("LeHoop and Ball Game", True, self.title_color)

        # get centered position
        self.title_rect = self.title_surf.get_rect(center=(self.screen_width // 2, 80))  # y=80 pixels from top


    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.button_rect.collidepoint(event.pos):
                return Game()  # switch to game

        return self  # stay in menu


    def update(self, dt):
        pass

    def scale_and_center_image(self, image):
        # get original size
        img_w, img_h = image.get_size()
    
        # compute scale factors
        scale_w = self.screen_width / img_w
        scale_h = self.screen_height / img_h
        scale = max(scale_w, scale_h)  # ensures no empty space

        # compute new size and scale image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        image = pygame.transform.scale(image, (new_w, new_h))

        # compute centered position
        x = (self.screen_width - new_w) // 2
        y = (self.screen_height - new_h) // 2

        return image, (x, y)
    



    def draw(self, screen):
        screen.blit(self.bliss_scaled, self.bliss_pos)
        screen.blit(self.lebron_scaled, self.lebron_pos)
        screen.blit(self.title_surf, self.title_rect)

        # Hover color
        mouse_pos = pygame.mouse.get_pos()
        if self.button_rect.collidepoint(mouse_pos):
            color = self.button_color
        else:
            color = self.button_hover_color

        pygame.draw.rect(screen, color, self.button_rect)

        text = self.font.render("PLAY", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.button_rect.center)
        screen.blit(text, text_rect)

class Game:
    
    def __init__(self):
        """All global variables involving the game menu should be put here"""

        self.W = 1000
        self.H = 650
        self.FPS = 120
        self.DT = 1.0 / 240.0

        #Defining game colors
        self.white = (245, 245, 245)
        self.black = (25, 25, 25)
        self.red = (220, 60, 60)
        self.blue = (50, 90, 220)
        self.gray = (180, 180, 180)
        self.green = (60, 190, 90)
        self.skytop = (235, 245, 255)
        self.skybot = (210, 230, 255)
        self.court = (245, 230, 210)
        self.courtline = (210, 190, 170)

        self.px_per_m = 220

        #Variables to run the game

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.bigfont = pygame.font.SysFont("consolas", 28, bold=True)


        self.g = 9.81
        self.L = 1.35
        self.A = 0.9
        self.wind_ax = 0.0

        self.green_fn = pygame.image.load("C:/Users/marti/Physics-Hackathon-2026/green_fn.png").convert_alpha()
        self.curry_moonshot = pygame.image.load("C:/Users/marti/Physics-Hackathon-2026/curry_moonshot.png").convert_alpha()

        self.score = 0
        self.level = 1

        self.trail = []  # list of recent pixel positions
        self.TRAIL_MAX = 28

        self.accumulator = 0.0
        self.running = True

        #Instatiating objects into Game()

        self.pend_dic = { 
            "pend_lvl1" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl2" : Pendulum(pivot_m=(1.2, -1.5), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl3" : Pendulum(pivot_m=(3, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl4" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl5" : Pendulum(pivot_m=(0, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl6" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl7" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl8" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl9" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3),
            "pend_lvl10" : Pendulum(pivot_m=(1.2, -1.0), L=self.L, A=self.A, g=self.g, phi=0.3)
            }

        self.pend = self.pend_dic.get("pend_lvl1")
        self.ball = Ball(radius_m=0.12)
        self.hoop = Hoop(center_m=(3.8, 2.15), radius_m=0.20)

        #Defining variable for image flash  
        self.im_green = False
        self.im_moonshot = False
        self.flash_start_time = 0
        self.flash_duration = 1000

        #Background variables

        self.background = bg.Background(self.W, self.H)

    #defining global functions into Game class ------

    def m2px(self, v):
        return v * self.px_per_m
    
    def world_to_screen(self, pos_m):
        return (int(self.m2px(pos_m[0])), int(self.m2px(pos_m[1])))
    
    def lerp(self, a, b, t):
        return a + (b - a) * t
    
    def draw_vertical_gradient(self, screen, top_color, bottom_color):
        # simple gradient: draw horizontal lines
        for y in range(self.H):
            t = y / (self.H - 1)
            c = (
                int(self.lerp(top_color[0], bottom_color[0], t)),
                int(self.lerp(top_color[1], bottom_color[1], t)),
                int(self.lerp(top_color[2], bottom_color[2], t)),
            )
            pygame.draw.line(screen, c, (0, y), (self.W, y))

    def aa_circle(self, screen, color, center, radius):
        # pygame's built-in draw is okay; gfxdraw is nicer if available
        try:
            import pygame.gfxdraw as gfx
            gfx.filled_circle(screen, center[0], center[1], radius, color)
            gfx.aacircle(screen, center[0], center[1], radius, color)
        except Exception:
            pygame.draw.circle(screen, color, center, radius)

    def aa_line(self, screen, color, p1, p2, width=2):
        # aaline is thin; for thicker lines draw multiple
        if width <= 1:
            pygame.draw.aaline(screen, color, p1, p2)
        else:
            pygame.draw.line(screen, color, p1, p2, width)

    def draw_text(self, screen, font, text, x, y, color=(25, 25, 25)):
        img = font.render(text, True, color)
        screen.blit(img, (x, y))

#global functions defined ---------------------------

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.pend.t = 0.0
                self.ball.attach_to(self.pend)
                self.trail.clear()
            if event.key == pygame.K_SPACE:
                if self.ball.attached:
                    self.ball.release_from(self.pend)
                    self.trail.clear()
            if event.key == pygame.K_ESCAPE:
                return Menu()
            if event.key == pygame.K_s:
                self.level += 1
        return self
    
    def update(self, frame_dt):
        self.background.update()
        self.accumulator += frame_dt  # add time passed this frame
        while self.accumulator >= self.DT:
            self.pend.step(self.DT)

            if self.ball.attached:
                self.ball.pos = self.pend.ball_pos()
            else:
                self.ball.step(self.DT, g=self.g, wind_ax=self.wind_ax)

                # score check
                if self.hoop.scored(self.ball):
                    self.score += 1
                    self.level += 1
                    self.pend = self.pend_dic[f"pend_lvl{self.level}"]
                    print(self.pend)
                    self.im_green = True
                    self.flash_start_time = pygame.time.get_ticks()

                    # difficulty ramp: wind after level 2
                    if self.level < 3:
                        self.wind_ax = 0.0

                    self.pend.t = 0.0
                    self.ball.attach_to(self.pend)
                    self.trail.clear()
                
                else: 
                    if self.ball.pos[1]>2.5 or self.ball.pos[0]<0 or self.ball.pos[0]>6:
                        self.im_moonshot = True
                        self.flash_start_time = pygame.time.get_ticks()

                # out of bounds reset
                if self.ball.pos[0] < -1 or self.ball.pos[0] > 6 or self.ball.pos[1] > 4:
                    self.pend.t = 0.0
                    self.ball.attach_to(self.pend)
                    self.trail.clear()

            self.accumulator -= self.DT
    
    def draw(self, screen):
        self.draw_vertical_gradient(screen, self.skytop, self.skybot)
        self.background.draw(screen)

        # court strip at bottom
        court_y = int(self.H * 0.78)
        pygame.draw.rect(screen, self.court, (0, court_y, self.W, self.H - court_y))
        pygame.draw.line(screen, self.courtline, (0, court_y), (self.W, court_y), 4)

        # draw hoop
        self.hoop.draw(screen)

        # draw pivot + rod
        pivot_px = self.world_to_screen(self.pend.pivot)
        ball_px = self.world_to_screen(self.ball.pos)
        self.aa_circle(screen, self.black, pivot_px, 6)
        self.aa_line(screen, (120, 120, 120), pivot_px, ball_px, width=5)

        # trail after release
        if not self.ball.attached:
            self.trail.append(ball_px)
            if len(self.trail) > self.TRAIL_MAX:
                self.trail.pop(0)
            for i in range(1, len(self.trail)):
                # fade by drawing lighter lines for older segments
                pygame.draw.line(screen, (140, 160, 200), self.trail[i-1], self.trail[i], 3)

        # ball shadow (fake depth)
        # more shadow when lower on screen
        shadow_strength = max(0.15, min(0.65, (ball_px[1] / self.H)))
        shadow_w = int(self.m2px(self.ball.r) * (1.6 + 0.8 * shadow_strength))
        shadow_h = int(self.m2px(self.ball.r) * (0.55 + 0.3 * shadow_strength))
        shadow_surf = pygame.Surface((shadow_w*2, shadow_h*2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, int(120 * shadow_strength)), shadow_surf.get_rect())
        screen.blit(shadow_surf, (ball_px[0] - shadow_w, court_y - shadow_h))

        # draw ball
        self.aa_circle(screen, self.blue, ball_px, int(self.m2px(self.ball.r)))
        self.aa_circle(screen, (240, 240, 255), (ball_px[0] - 6, ball_px[1] - 6), max(2, int(self.m2px(self.ball.r) * 0.35)))

        # HUD
        self.draw_text(screen, self.font, f"L={self.pend.L:.2f} m   A={self.pend.A:.2f} rad   g={self.g:.2f}", 20, 18)
        self.draw_text(screen, self.font, "SPACE=release   R=reset  S=skip level", 20, 62)
        self.draw_text(screen, self.font, f"x position ball: {self.ball.pos[0]:.2f} m", 20, 84)
        self.draw_text(screen, self.font, f"y position ball: {self.ball.pos[1]:.2f} m", 20, 106) 

        title = self.bigfont.render(f"Score: {self.score}   Level: {self.level}", True, (30, 30, 30))
        screen.blit(title, (self.W - title.get_width() - 20, 16))

        if self.im_green:
            elapsed = pygame.time.get_ticks() - self.flash_start_time
            alpha = max(255 - 255 * (elapsed / self.flash_duration), 0)

            image_copy = self.green_fn.copy()
            image_copy.set_alpha(int(alpha))
            screen.blit(image_copy, (0, 0))

            if alpha <= 0:
                self.im_green = False  # flash finished

        if self.im_moonshot: #and ball.pos[1]>2.5:
            elapsed = pygame.time.get_ticks() - self.flash_start_time
            alpha = max(255 - 255 * (elapsed / self.flash_duration), 0)

            image_copy = self.curry_moonshot.copy()
            image_copy.set_alpha(int(alpha))
            screen.blit(image_copy, (0, 0))

            if alpha <= 0:
                self.im_moonshot = False  # flash finished


# ----------------- main loop -----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((1000, 650))  # only once
    pygame.display.set_caption("Pendulum Basketball (Upgraded Look)")

    my_game = Game()
    my_menu = Menu()
    current_screen = my_menu
    while my_game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                my_game.running = False
            current_screen = current_screen.handle_events(event)
            


        frame_dt = my_game.clock.tick(my_game.FPS) / 1000.0

        current_screen.update(frame_dt)
        
        current_screen.draw(screen)
        
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
