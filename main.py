"""Physics Hoop — entry point.

A small state machine over the presentation layer:

    MENU -> GAME            (Play, built-in ladder)
    MENU -> MAPS            (Map Gallery)
    MAPS -> GAME            (play a saved run)
    MAPS -> NEWMAP -> GAME  (build a run, Test, then Back to gallery)
    GAME -> MENU            (M or ESC-to-menu)

Run with:  python main.py
"""
from __future__ import annotations

import sys


import pygame

from game.config import FPS, WINDOW_SCALE, MAP_DIR
from game.engine import Game, builtin_run
from game.maps import load_run
from game.menu import About, MapEditor, MapGallery, Menu
from game.ui import draw_hud


def window_size():
    sizes = pygame.display.get_desktop_sizes()
    if isinstance(sizes, (list, tuple)) and sizes and isinstance(sizes[0], (list, tuple)):
        W, H = sizes[0]  # first display
    else:
        W, H = sizes
    W, H = int(W * WINDOW_SCALE), int(H * WINDOW_SCALE)
    return max(W, 860), max(H, 560)


def main():
    pygame.init()
    # audio: mono 16-bit (matches the synthesized cues in game.sound)
    try:
        pygame.mixer.init(22050, -16, 1)
    except pygame.error:
        pass
    pygame.display.set_caption("Physics Hoop")
    W, H = window_size()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    state = "MENU"
    menu = Menu(W, H, screen)
    gallery = MapGallery(W, H, screen)
    about = About(W, H, screen)
    editor = None
    game = None
    editor_return = "MAPS"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if state == "MENU":
                r = menu.handle(event)
                if r == "GAME":
                    game = Game(W, H, screen, run=builtin_run())
                    state = "GAME"
                elif r == "MAPS":
                    gallery.refresh()
                    state = "MAPS"
                elif r == "ABOUT":
                    state = "ABOUT"
                elif r == "QUIT":
                    running = False

            elif state == "ABOUT":
                if about.handle(event) == "MENU":
                    state = "MENU"

            elif state == "MAPS":
                r = gallery.handle(event)
                if r is None:
                    pass
                elif r == "MENU":
                    state = "MENU"
                elif r == "NEWMAP":
                    editor = MapEditor(W, H, screen, name="run_" + str(len(gallery.maps) + 1))
                    editor_return = "MAPS"
                    state = "NEWMAP"
                elif isinstance(r, tuple) and r[0] == "PLAYMAP":
                    name = r[1]
                    levels = load_run(name, MAP_DIR)
                    game = Game(W, H, screen, run=levels, run_name=name)
                    state = "GAME"

            elif state == "NEWMAP":
                assert editor is not None
                r = editor.handle(event)
                if r is None:
                    pass
                elif r == "SAVED":
                    gallery.refresh()
                    state = editor_return
                elif r == "MENU":
                    state = "MENU"
                elif isinstance(r, tuple) and r[0] == "TEST":
                    game = Game(W, H, screen, run=r[1], run_name="test")
                    state = "GAME"

            elif state == "GAME":
                r = game.handle(event)
                if r == "MENU":
                    state = "MENU"

        if state == "MENU":
            menu.draw()
        elif state == "ABOUT":
            about.draw()
        elif state == "MAPS":
            gallery.draw()
        elif state == "NEWMAP":
            assert editor is not None
            editor.tick(clock.tick(FPS) / 1000.0)   # advance the live preview
            editor.draw()
        elif state == "GAME":
            assert game is not None
            game.update(clock.tick(FPS) / 1000.0)
            game.draw()
            draw_hud(screen, game, W, H)
        else:
            clock.tick(FPS)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
