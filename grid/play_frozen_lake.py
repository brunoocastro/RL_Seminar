"""
Interactive Frozen Lake game using Pygame.
Use arrow keys or WASD to move. Press Q to quit.
"""
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import pygame
import sys

# Disable output buffering
import os
os.environ['PYTHONUNBUFFERED'] = '1'


class FrozenLakeGame:
    """Simple Pygame-based Frozen Lake game with visual feedback."""
    
    def __init__(self, map_size=4, is_slippery=False):
        """Initialize the game."""
        pygame.init()
        
        # Game settings
        self.map_size = map_size
        self.is_slippery = is_slippery
        self.cell_size = 80
        self.info_width = 300
        
        # Window setup
        self.window_width = self.map_size * self.cell_size + self.info_width
        self.window_height = self.map_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Frozen Lake Game")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (135, 206, 250)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)
        self.YELLOW = (255, 255, 100)
        self.GRAY = (200, 200, 200)
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)
        
        # Create environment
        if map_size == 4:
            self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
        else:
            custom_map = generate_random_map(size=map_size, p=0.8)
            self.env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=is_slippery)
        
        # Get the map description
        self.map_desc = self.env.unwrapped.desc
        
        # Game state
        self.wins = 0
        self.losses = 0
        self.total_games = 0
        self.current_state = None
        self.game_over = False
        self.clock = pygame.time.Clock()
    
    def get_position(self, state):
        """Convert state number to (row, col) position."""
        row = state // self.map_size
        col = state % self.map_size
        return row, col
    
    def draw_grid(self):
        """Draw the Frozen Lake grid."""
        for row in range(self.map_size):
            for col in range(self.map_size):
                x = col * self.cell_size
                y = row * self.cell_size
                
                # Get cell type (decode from numpy bytes)
                cell = self.map_desc[row][col].decode('utf-8')
                
                # Choose color
                if cell == 'S':
                    color = self.GREEN
                elif cell == 'F':
                    color = self.BLUE
                elif cell == 'H':
                    color = self.RED
                elif cell == 'G':
                    color = self.YELLOW
                else:
                    color = self.WHITE
                
                # Draw cell
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.BLACK, (x, y, self.cell_size, self.cell_size), 2)
                
                # Draw letter
                text = self.font.render(cell, True, self.BLACK)
                text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                self.screen.blit(text, text_rect)
    
    def draw_agent(self):
        """Draw the agent at current position."""
        if self.current_state is not None:
            row, col = self.get_position(self.current_state)
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.BLACK, (x, y), 20, 5)
    
    def draw_info(self):
        """Draw game information on the right side."""
        info_x = self.map_size * self.cell_size + 20
        
        # Background for info panel
        pygame.draw.rect(self.screen, self.GRAY, 
                        (self.map_size * self.cell_size, 0, self.info_width, self.window_height))
        
        # Title
        y = 20
        title = self.font.render("FROZEN LAKE", True, self.BLACK)
        self.screen.blit(title, (info_x, y))
        
        # Mode
        y += 50
        mode_text = "Slippery" if self.is_slippery else "Normal"
        mode = self.small_font.render(f"Mode: {mode_text}", True, self.BLACK)
        self.screen.blit(mode, (info_x, y))
        
        # Statistics
        y += 60
        stats_title = self.font.render("STATISTICS", True, self.BLACK)
        self.screen.blit(stats_title, (info_x, y))
        
        y += 40
        games_text = self.small_font.render(f"Games: {self.total_games}", True, self.BLACK)
        self.screen.blit(games_text, (info_x, y))
        
        y += 35
        wins_text = self.small_font.render(f"Wins: {self.wins}", True, self.GREEN)
        self.screen.blit(wins_text, (info_x, y))
        
        y += 35
        losses_text = self.small_font.render(f"Losses: {self.losses}", True, self.RED)
        self.screen.blit(losses_text, (info_x, y))
        
        # Win rate
        if self.total_games > 0:
            y += 35
            win_rate = (self.wins / self.total_games) * 100
            rate_text = self.small_font.render(f"Win Rate: {win_rate:.1f}%", True, self.BLACK)
            self.screen.blit(rate_text, (info_x, y))
        
        # Controls
        y += 80
        controls_title = self.small_font.render("CONTROLS", True, self.BLACK)
        self.screen.blit(controls_title, (info_x, y))
        
        y += 35
        wasd = self.small_font.render("WASD or Arrows", True, self.BLACK)
        self.screen.blit(wasd, (info_x, y))
        
        y += 30
        quit_text = self.small_font.render("Q = Quit", True, self.BLACK)
        self.screen.blit(quit_text, (info_x, y))
        
        y += 30
        restart_text = self.small_font.render("R = Restart", True, self.BLACK)
        self.screen.blit(restart_text, (info_x, y))
        
        # Game over message
        if self.game_over:
            y += 60
            if hasattr(self, 'last_reward') and self.last_reward > 0:
                msg = self.font.render("YOU WON!", True, self.GREEN)
            else:
                msg = self.font.render("GAME OVER", True, self.RED)
            self.screen.blit(msg, (info_x, y))
    
    def draw(self):
        """Draw everything on screen."""
        self.screen.fill(self.WHITE)
        self.draw_grid()
        self.draw_agent()
        self.draw_info()
        pygame.display.flip()
    
    def reset_game(self):
        """Reset to a new game."""
        self.current_state, _ = self.env.reset()
        self.game_over = False
        self.total_games += 1
        self.draw()
    
    def handle_action(self, action):
        """Execute an action in the environment."""
        if not self.game_over:
            observation, reward, terminated, truncated, _ = self.env.step(action)
            self.current_state = observation
            self.last_reward = reward
            
            if terminated or truncated:
                self.game_over = True
                if reward > 0:
                    self.wins += 1
                else:
                    self.losses += 1
            
            self.draw()
    
    def run(self):
        """Main game loop."""
        self.reset_game()
        running = True
        
        # Key mapping
        action_keys = {
            pygame.K_LEFT: 0,   pygame.K_a: 0,  # LEFT
            pygame.K_DOWN: 1,   pygame.K_s: 1,  # DOWN
            pygame.K_RIGHT: 2,  pygame.K_d: 2,  # RIGHT
            pygame.K_UP: 3,     pygame.K_w: 3,  # UP
        }
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    # Quit
                    if event.key == pygame.K_q:
                        running = False
                    
                    # Restart
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    
                    # Move actions
                    elif event.key in action_keys and not self.game_over:
                        action = action_keys[event.key]
                        self.handle_action(action)
                    
                    # Auto-restart if game over
                    elif self.game_over and event.key in action_keys:
                        self.reset_game()
            
            self.clock.tick(60)  # 60 FPS
        
        self.env.close()
        pygame.quit()


def main():
    """Main entry point."""
    print("\n🎮 FROZEN LAKE GAME SETUP")
    print("=" * 50)
    
    # Choose map size
    while True:
        try:
            map_size_input = input("\nMap size (4, 8, or Enter for 4): ").strip()
            map_size = int(map_size_input) if map_size_input else 4
            if map_size in [4, 8]:
                break
            print("Please choose 4 or 8")
        except ValueError:
            print("Invalid input! Use 4 or 8")
    
    # Choose mode
    slippery_input = input("\nSlippery mode? (Y/N, Enter for N): ").strip().lower()
    is_slippery = slippery_input == 'y'
    
    # Start the game
    print("\n🎮 Starting game window...")
    print("Controls: WASD or Arrow Keys to move, Q to quit, R to restart")
    
    game = FrozenLakeGame(map_size=map_size, is_slippery=is_slippery)
    game.run()
    
    print("\n👋 Thanks for playing!")


if __name__ == "__main__":
    main()