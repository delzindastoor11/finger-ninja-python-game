import pygame  # Main game library
import cv2  # Computer vision for camera
import mediapipe as mp  # Hand tracking
import random  # For random positions and speeds
import os  # File path operations
import math  # Mathematical functions
import time  # Time operations

WIDTH, HEIGHT = 1280, 720  # Game window dimensions
FPS = 60  # Frames per second
BG_COLOR = (0, 0, 0)  # Black background fallback
FONT_COLOR = (255, 255, 255)  # White text color
GRAVITY = 0.4  # Physics gravity constant
INITIAL_FRUIT_SPEED_Y = -25  # Initial upward velocity
INITIAL_FRUIT_SPEED_X_RANGE = (-5, 5)  # Horizontal speed range
SPAWN_INTERVAL = 1000  # Object spawn interval in ms

FRUIT_SIZE = (100, 100)  # Fruit dimensions
BOMB_SIZE = (100, 100)  # Bomb dimensions
EXPLOSION_SIZE = (150, 150)  # Explosion dimensions

pygame.init()  # Initialize pygame
pygame.mixer.init()  # Initialize sound system
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create game window
pygame.display.set_caption("AI Fruit Ninja")  # Set window title
clock = pygame.time.Clock()  # it is a control frame rate for your game

ASSET_DIR = "assets"  # Main asset directory
FRUIT_DIR = os.path.join(ASSET_DIR, "fruits")  # Fruit images folder
BOMB_DIR = os.path.join(ASSET_DIR, "bombs")  # Bomb images folder
SOUND_DIR = os.path.join(ASSET_DIR, "sounds")  # Sound files folder

def load_and_scale_image(directory, filename, size, alpha=True):  # Load and resize image
    path = os.path.join(directory, filename)  # Create full file path
    try:  # Try to load image
        if alpha:  # If alpha channel needed
            image = pygame.image.load(path).convert_alpha()  # Load with transparency
        else:  # If no alpha needed
            image = pygame.image.load(path).convert()  # Load without transparency
        return pygame.transform.scale(image, size)  # Scale to desired size
    except pygame.error as e:  # If loading fails
        print(f"Error loading image {path}: {e}")  # Print error message
        return None  # Return None if failed

BACKGROUND_IMAGE = load_and_scale_image(ASSET_DIR, "background.jpg", (WIDTH, HEIGHT), alpha=False)  # Load background
if BACKGROUND_IMAGE is None:  # If background not found
    print(f"Warning: background.jpg not found in {ASSET_DIR}. Using black background.")  # Warning message

SLICE_SOUND = None  # Initialize slice sound variable
BOMB_SOUND = None  # Initialize bomb sound variable
try:  # Try to load sounds
    SLICE_SOUND = pygame.mixer.Sound(os.path.join(SOUND_DIR, "slice.mp3"))  # Load slice sound
    BOMB_SOUND = pygame.mixer.Sound(os.path.join(SOUND_DIR, "bomb.mp3"))  # Load bomb sound
except pygame.error:  # If sounds fail to load
    print(f"Warning: Sound files not found in {SOUND_DIR}. Slicing/bomb sounds will be silent.")  # Warning message

FRUIT_ASSETS_MAP = {  # Fruit asset mapping
    "apple": {"img": "apple.png", "cut_img": "apple_cut.png"},  # Apple images
    "banana": {"img": "banana.png", "cut_img": "banana_cut.png"},  # Banana images
    "orange": {"img": "orange.png", "cut_img": "orange_cut.png"},  # Orange images
}

BOMB_ASSETS_MAP = {  # Bomb asset mapping
    "bomb": {"img": "bomb.png", "explosion_img": "bomb_explosion.png"}  # Bomb images
}

class AppState:  # Game state constants
    MENU = 0  # Main menu state
    PLAYING = 1  # Game playing state
    PAUSED = 2  # Game paused state
    GAME_OVER = 3  # Game over state

class HandTracker:  # Hand tracking class
    def __init__(self):  # Initialize hand tracker
        self.mp_hands = mp.solutions.hands  # MediaPipe hands solution
        self.hands = self.mp_hands.Hands(  # Configure hand detection
            static_image_mode=False,  # Process video stream
            max_num_hands=1,  # Track one hand
            min_detection_confidence=0.7,  # Detection threshold
            min_tracking_confidence=0.5  # Tracking threshold
        )
        self.mp_draw = mp.solutions.drawing_utils  # Drawing utilities
        self.cap = cv2.VideoCapture(0)  # Initialize camera
        
        self.index_finger_tip = None  # Current finger position
        self.lm_history = []  # Position history for slicing
        self.history_length = 15  # History buffer size
        
        self.smoothing_alpha = 0.2  # Smoothing factor
        self.smoothed_tip_x = -1  # Smoothed X coordinate
        self.smoothed_tip_y = -1  # Smoothed Y coordinate
        
        self.hand_landmarks = None  # Current hand landmarks

    def process_frame(self):  # Process camera frame
        success, img = self.cap.read()  # Read frame from camera
        if not success:  # If frame read fails
            print("Error: Could not access webcam. Please ensure it's connected and not in use.")  # Error message
            return None, None  # Return None values

        img = cv2.flip(img, 1)  # Mirror image horizontally
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = self.hands.process(img_rgb)  # Process for hand detection

        self.index_finger_tip = None  # Reset finger position
        self.hand_landmarks = None  # Reset landmarks
        
        if results.multi_hand_landmarks:  # If hands detected
            for hand_landmarks in results.multi_hand_landmarks:  # For each hand
                self.hand_landmarks = hand_landmarks  # Store landmarks

                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)  # Draw hand skeleton

                norm_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x  # Get normalized X
                norm_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y  # Get normalized Y
                
                scaled_tip_x_raw = int(norm_x * WIDTH)  # Scale X to window width
                scaled_tip_y_raw = int(norm_y * HEIGHT)  # Scale Y to window height

                if self.smoothed_tip_x == -1:  # First detection
                    self.smoothed_tip_x = scaled_tip_x_raw  # Initialize smoothed X
                    self.smoothed_tip_y = scaled_tip_y_raw  # Initialize smoothed Y
                else:  # Apply smoothing filter
                    self.smoothed_tip_x = self.smoothing_alpha * scaled_tip_x_raw + (1 - self.smoothing_alpha) * self.smoothed_tip_x  # Smooth X
                    self.smoothed_tip_y = self.smoothing_alpha * scaled_tip_y_raw + (1 - self.smoothing_alpha) * self.smoothed_tip_y  # Smooth Y
                
                self.index_finger_tip = (int(self.smoothed_tip_x), int(self.smoothed_tip_y))  # Set finger position

                h_img, w_img, c_img = img.shape  # Get image dimensions
                cv2.circle(img, (int(norm_x * w_img), int(norm_y * h_img)), 10, (0, 255, 0), cv2.FILLED)  # Draw finger indicator

        if self.index_finger_tip:  # If finger detected
            self.lm_history.append(self.index_finger_tip)  # Add to history
            if len(self.lm_history) > self.history_length:  # If history too long
                self.lm_history.pop(0)  # Remove oldest entry
        else:  # If no finger detected
            self.lm_history = []  # Clear history
            self.smoothed_tip_x = -1  # Reset smoothing
            self.smoothed_tip_y = -1  # Reset smoothing

        return img, self.index_finger_tip  # Return image and finger position

    def is_hand_open(self):  # Check if hand is open
        if not self.hand_landmarks:  # If no landmarks
            return False  # Hand not open

        landmarks = self.hand_landmarks.landmark  # Get landmarks
        Y_EXTEND_THRESHOLD = 0.04  # Extension threshold

        fingers_extended = []  # List of finger states
        
        fingers_extended.append(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y - Y_EXTEND_THRESHOLD)  # Check index finger
        fingers_extended.append(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y - Y_EXTEND_THRESHOLD)  # Check middle finger
        fingers_extended.append(landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP].y - Y_EXTEND_THRESHOLD)  # Check ring finger
        fingers_extended.append(landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < landmarks[self.mp_hands.HandLandmark.PINKY_MCP].y - Y_EXTEND_THRESHOLD)  # Check pinky finger
        fingers_extended.append(landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y < landmarks[self.mp_hands.HandLandmark.THUMB_IP].y - Y_EXTEND_THRESHOLD)  # Check thumb
        
        return all(fingers_extended)  # Return true if all fingers extended

    def is_index_finger_pointed(self):  # Check if index finger pointed
        if not self.hand_landmarks:  # If no landmarks
            return False  # Not pointing

        landmarks = self.hand_landmarks.landmark  # Get landmarks
        
        EXTEND_THRESHOLD = 0.02  # Extension threshold
        CURL_THRESHOLD = 0.02  # Curl threshold

        index_extended = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y - EXTEND_THRESHOLD  # Check index extended

        middle_curled = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y + CURL_THRESHOLD  # Check middle curled
        ring_curled = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y + CURL_THRESHOLD  # Check ring curled
        pinky_curled = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y > landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y + CURL_THRESHOLD  # Check pinky curled
        thumb_curled = landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y > landmarks[self.mp_hands.HandLandmark.THUMB_IP].y + CURL_THRESHOLD  # Check thumb curled
        
        return index_extended and middle_curled and ring_curled and pinky_curled and thumb_curled  # Return pointing state

    def release(self):  # Release resources
        self.cap.release()  # Release camera
        cv2.destroyAllWindows()  # Close OpenCV windows

class Fruit(pygame.sprite.Sprite):  # Fruit game object
    def __init__(self, name, image_filename, cut_image_filename):  # Initialize fruit
        super().__init__()  # Call parent constructor
        self.name = name  # Fruit name
        
        self.original_image = load_and_scale_image(FRUIT_DIR, image_filename, FRUIT_SIZE)  # Load fruit image
        if self.original_image is None:  # If image not found
            self.original_image = pygame.Surface(FRUIT_SIZE, pygame.SRCALPHA)  # Create placeholder surface
            pygame.draw.rect(self.original_image, (255, 0, 0, 150), (0,0,FRUIT_SIZE[0],FRUIT_SIZE[1]), border_radius=10)  # Draw red rectangle
            pygame.draw.circle(self.original_image, (0,0,0), (FRUIT_SIZE[0]//2, FRUIT_SIZE[1]//2), FRUIT_SIZE[0]//4)  # Draw seed
            print(f"Using placeholder for fruit: {image_filename}")  # Warning message

        self.cut_image = load_and_scale_image(FRUIT_DIR, cut_image_filename, FRUIT_SIZE)  # Load cut fruit image
        if self.cut_image is None:  # If cut image not found
            self.cut_image = pygame.Surface(FRUIT_SIZE, pygame.SRCALPHA)  # Create placeholder surface
            pygame.draw.rect(self.cut_image, (150, 150, 150, 150), (0,0,FRUIT_SIZE[0],FRUIT_SIZE[1]), border_radius=10)  # Draw grey rectangle
            print(f"Using placeholder for cut fruit: {cut_image_filename}")  # Warning message

        self.image = self.original_image  # Set current image
        self.rect = self.image.get_rect()  # Get image rectangle

        self.rect.x = random.randint(50, WIDTH - 150)  # Random X position
        self.rect.y = HEIGHT + 50  # Start below screen

        self.vel_x = random.randint(INITIAL_FRUIT_SPEED_X_RANGE[0], INITIAL_FRUIT_SPEED_X_RANGE[1])  # Random horizontal velocity
        self.vel_y = INITIAL_FRUIT_SPEED_Y  # Initial upward velocity
        self.rotation_speed = random.uniform(-5, 5)  # Random rotation speed
        self.angle = 0  # Initial rotation angle

        self.sliced = False  # Slice state
        self.slice_time = 0  # Time since sliced
        self.fade_start_time = FPS * 1  # Fade start time
        self.fade_duration = FPS * 0.5  # Fade duration

    def update(self):  # Update fruit state
        if not self.sliced:  # If not sliced
            self.vel_y += GRAVITY  # Apply gravity
            self.rect.x += self.vel_x  # Update X position
            self.rect.y += self.vel_y  # Update Y position

            if self.rect.left < 0 or self.rect.right > WIDTH:  # If hitting sides
                self.vel_x *= -1  # Reverse horizontal velocity
                if self.rect.left < 0:  # If past left edge
                    self.rect.left = 0  # Clamp to left
                if self.rect.right > WIDTH:  # If past right edge
                    self.rect.right = WIDTH  # Clamp to right

            self.angle = (self.angle + self.rotation_speed) % 360  # Update rotation
            self.image = pygame.transform.rotate(self.original_image, self.angle)  # Rotate image
            self.rect = self.image.get_rect(center=self.rect.center)  # Update rect center

        else:  # If sliced
            self.vel_y += GRAVITY * 1.5  # Fall faster when sliced
            self.rect.x += self.vel_x  # Update X position
            self.rect.y += self.vel_y  # Update Y position
            self.slice_time += 1  # Increment slice timer

            if self.slice_time > self.fade_start_time:  # If fade should start
                time_since_fade_start = self.slice_time - self.fade_start_time  # Calculate fade progress
                if self.fade_duration > 0:  # If fade duration valid
                    alpha = 255 - int(255 * (time_since_fade_start / self.fade_duration))  # Calculate alpha
                    alpha = max(0, alpha)  # Clamp alpha to minimum 0

                    if self.image.get_alpha() != alpha:  # If alpha changed
                        temp_image = self.cut_image.copy()  # Copy cut image
                        temp_image.set_alpha(alpha)  # Set new alpha
                        self.image = temp_image  # Update current image
                
            if self.slice_time > self.fade_start_time + self.fade_duration:  # If fade complete
                self.kill()  # Remove sprite

    def slice(self):  # Slice the fruit
        if not self.sliced:  # If not already sliced
            self.sliced = True  # Mark as sliced
            self.image = self.cut_image  # Switch to cut image
            return True  # Return success
        return False  # Return failure if already sliced

class Bomb(pygame.sprite.Sprite):  # Bomb game object
    def __init__(self, image_filename, explosion_image_filename):  # Initialize bomb
        super().__init__()  # Call parent constructor
        
        self.original_image = load_and_scale_image(BOMB_DIR, image_filename, BOMB_SIZE)  # Load bomb image
        if self.original_image is None:  # If image not found
            self.original_image = pygame.Surface(BOMB_SIZE, pygame.SRCALPHA)  # Create placeholder surface
            pygame.draw.circle(self.original_image, (100, 100, 100, 200), (BOMB_SIZE[0]//2, BOMB_SIZE[1]//2), BOMB_SIZE[0]//2)  # Draw grey circle
            pygame.draw.line(self.original_image, (255, 255, 0), (BOMB_SIZE[0]//2, 0), (BOMB_SIZE[0]//2 + 20, 20), 5)  # Draw fuse
            print(f"Using placeholder for bomb: {image_filename}")  # Warning message
        
        self.explosion_image = load_and_scale_image(BOMB_DIR, explosion_image_filename, EXPLOSION_SIZE)  # Load explosion image
        if self.explosion_image is None:  # If explosion image not found
            self.explosion_image = pygame.Surface(EXPLOSION_SIZE, pygame.SRCALPHA)  # Create placeholder surface
            pygame.draw.circle(self.explosion_image, (255, 165, 0, 200), (EXPLOSION_SIZE[0]//2, EXPLOSION_SIZE[1]//2), EXPLOSION_SIZE[0]//2)  # Draw orange circle
            pygame.draw.circle(self.explosion_image, (255, 255, 0, 150), (EXPLOSION_SIZE[0]//2, EXPLOSION_SIZE[1]//2), EXPLOSION_SIZE[0]//4)  # Draw yellow center
            print(f"Using placeholder for explosion: {explosion_image_filename}")  # Warning message

        self.image = self.original_image  # Set current image
        self.rect = self.image.get_rect()  # Get image rectangle

        self.rect.x = random.randint(50, WIDTH - 150)  # Random X position
        self.rect.y = HEIGHT + 50  # Start below screen

        self.vel_x = random.randint(INITIAL_FRUIT_SPEED_X_RANGE[0], INITIAL_FRUIT_SPEED_X_RANGE[1])  # Random horizontal velocity
        self.vel_y = INITIAL_FRUIT_SPEED_Y  # Initial upward velocity

        self.exploded = False  # Explosion state
        self.explosion_time = 0  # Time since exploded
        self.fade_start_time = FPS * 0.2  # Fade start time
        self.fade_duration = FPS * 0.3  # Fade duration

    def update(self):  # Update bomb state
        if not self.exploded:  # If not exploded
            self.vel_y += GRAVITY  # Apply gravity
            self.rect.x += self.vel_x  # Update X position
            self.rect.y += self.vel_y  # Update Y position

            if self.rect.left < 0 or self.rect.right > WIDTH:  # If hitting sides
                self.vel_x *= -1  # Reverse horizontal velocity
                if self.rect.left < 0:  # If past left edge
                    self.rect.left = 0  # Clamp to left
                if self.rect.right > WIDTH:  # If past right edge
                    self.rect.right = WIDTH  # Clamp to right
        else:  # If exploded
            self.explosion_time += 1  # Increment explosion timer

            if self.explosion_time > self.fade_start_time:  # If fade should start
                time_since_fade_start = self.explosion_time - self.fade_start_time  # Calculate fade progress
                if self.fade_duration > 0:  # If fade duration valid
                    alpha = 255 - int(255 * (time_since_fade_start / self.fade_duration))  # Calculate alpha
                    alpha = max(0, alpha)  # Clamp alpha to minimum 0

                    if self.image.get_alpha() != alpha:  # If alpha changed
                        temp_image = self.explosion_image.copy()  # Copy explosion image
                        temp_image.set_alpha(alpha)  # Set new alpha
                        self.image = temp_image  # Update current image

            if self.explosion_time > self.fade_start_time + self.fade_duration:  # If fade complete
                self.kill()  # Remove sprite

    def explode(self):  # Explode the bomb
        if not self.exploded:  # If not already exploded
            self.exploded = True  # Mark as exploded
            self.image = self.explosion_image  # Switch to explosion image
            self.rect = self.image.get_rect(center=self.rect.center)  # Update rect center
            global game_state  # Access global game state
            game_state = AppState.GAME_OVER  # Set game over
            return True  # Return success
        return False  # Return failure if already exploded

def draw_text(surface, text, size, x, y, color=FONT_COLOR):  # Draw text on screen
    font = pygame.font.Font(None, size)  # Create font
    text_surface = font.render(text, True, color)  # Render text
    text_rect = text_surface.get_rect(center=(x, y))  # Get text rectangle
    surface.blit(text_surface, text_rect)  # Draw text to surface

def reset_game():  # Reset game state
    global score, lives, game_state, last_spawn_time, slice_path, last_finger_pos  # Access globals
    score = 0  # Reset score
    lives = 5  # Reset lives
    game_state = AppState.PLAYING  # Set to playing
    all_sprites.empty()  # Clear all sprites
    fruits_and_bombs.empty()  # Clear active objects
    last_spawn_time = pygame.time.get_ticks()  # Reset spawn timer
    slice_path = []  # Clear slice path
    last_finger_pos = None  # Reset finger position

def spawn_object():  # Spawn new game object
    obj_type = random.choices(["fruit", "bomb"], weights=[0.8, 0.2], k=1)[0]  # Choose object type
    
    if obj_type == "fruit":  # If spawning fruit
        fruit_name = random.choice(list(FRUIT_ASSETS_MAP.keys()))  # Choose random fruit
        fruit_data = FRUIT_ASSETS_MAP[fruit_name]  # Get fruit data
        new_fruit = Fruit(fruit_name, fruit_data["img"], fruit_data["cut_img"])  # Create fruit
        all_sprites.add(new_fruit)  # Add to all sprites
        fruits_and_bombs.add(new_fruit)  # Add to active objects
    else:  # If spawning bomb
        bomb_data = BOMB_ASSETS_MAP["bomb"]  # Get bomb data
        new_bomb = Bomb(bomb_data["img"], bomb_data["explosion_img"])  # Create bomb
        all_sprites.add(new_bomb)  # Add to all sprites
        fruits_and_bombs.add(new_bomb)  # Add to active objects

def line_rect_intersection(p1, p2, rect):  # Check line-rectangle intersection
    if rect.collidepoint(p1) or rect.collidepoint(p2):  # If either point inside rect
        return True  # Intersection found

    def ccw(A, B, C):  # Counter-clockwise test
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])  # Cross product test

    def intersect(A, B, C, D):  # Line segment intersection test
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)  # Check both segments

    rect_points = [rect.topleft, rect.topright, rect.bottomright, rect.bottomleft]  # Get rectangle corners
    for i in range(4):  # For each edge
        r_p1 = rect_points[i]  # First corner
        r_p2 = rect_points[(i + 1) % 4]  # Next corner
        if intersect(p1, p2, r_p1, r_p2):  # If line intersects edge
            return True  # Intersection found
    return False  # No intersection

score = 0  # Current score
lives = 5  # Current lives
game_state = AppState.MENU  # Current game state
all_sprites = pygame.sprite.Group()  # All game sprites
fruits_and_bombs = pygame.sprite.Group()  # Active game objects

tracker = HandTracker()  # Initialize hand tracker
last_finger_pos = None  # Previous finger position
slice_path = []  # Finger movement path
SLICE_PATH_LENGTH = 15  # Path length for slicing
SLICE_MIN_DISTANCE = 60  # Minimum slice distance

last_open_hand_time = 0  # Last open hand gesture time
open_hand_debounce_ms = 1000  # Gesture debounce time

BASE_CURSOR_RADIUS = 15  # Base cursor size
PULSE_AMPLITUDE = 5  # Pulse animation amplitude
PULSE_SPEED = 0.01  # Pulse animation speed

SLICER_COLOR = (0, 191, 255)  # Slicer line color
SLICER_WIDTH = 8  # Slicer line thickness

TRAIL_PARTICLE_COUNT = 8  # Trail particle count
TRAIL_BASE_RADIUS = 5  # Trail particle radius
TRAIL_COLOR = (255, 200, 0)  # Trail particle color

running = True  # Main loop flag
last_spawn_time = pygame.time.get_ticks()  # Last object spawn time

while running:  # Main game loop
    for event in pygame.event.get():  # Process events
        if event.type == pygame.QUIT:  # If quit event
            running = False  # Exit loop
        if event.type == pygame.KEYDOWN:  # If key pressed
            if event.key == pygame.K_r and game_state == AppState.GAME_OVER:  # R key in game over
                reset_game()  # Reset game
            if event.key == pygame.K_s and game_state == AppState.MENU:  # S key in menu
                game_state = AppState.PLAYING  # Start playing
            if event.key == pygame.K_p and game_state == AppState.PLAYING:  # P key while playing
                game_state = AppState.PAUSED  # Pause game

    img_cv2, current_finger_pos = tracker.process_frame()  # Process camera frame
    current_time_ms = pygame.time.get_ticks()  # Get current time

    if img_cv2 is None:  # If camera failed
        screen.fill(BG_COLOR)  # Fill screen black
        draw_text(screen, "Webcam Error: Please check connection.", 50, WIDTH // 2, HEIGHT // 2, (255, 0, 0))  # Show error
        pygame.display.flip()  # Update display
        clock.tick(FPS)  # Maintain framerate
        continue  # Skip frame

    if BACKGROUND_IMAGE:  # If background image exists
        screen.blit(BACKGROUND_IMAGE, (0, 0))  # Draw background
    else:  # If no background
        screen.fill(BG_COLOR)  # Fill with solid color

    if game_state == AppState.MENU:  # If in menu state
        draw_text(screen, "AI FRUIT NINJA", 100, WIDTH // 2, HEIGHT // 3, (0, 255, 255))  # Draw title
        draw_text(screen, "Open Palm to Start", 60, WIDTH // 2, HEIGHT // 2 + 50)  # Draw instruction
        draw_text(screen, "Slice fruits, avoid bombs!", 40, WIDTH // 2, HEIGHT // 2 + 150)  # Draw description
        
        if tracker.is_hand_open() and (current_time_ms - last_open_hand_time > open_hand_debounce_ms):  # If open hand detected
            game_state = AppState.PLAYING  # Start game
            reset_game()  # Initialize game
            last_open_hand_time = current_time_ms  # Update gesture time

    elif game_state == AppState.PLAYING:  # If playing
        if current_finger_pos:  # If finger detected
            pulse_radius = BASE_CURSOR_RADIUS + PULSE_AMPLITUDE * math.sin(current_time_ms * PULSE_SPEED)  # Calculate pulse radius
            
            pygame.draw.circle(screen, (0, 255, 255), current_finger_pos, int(pulse_radius), 3)  # Draw finger cursor

            slice_path.append(current_finger_pos)  # Add to slice path
            if len(slice_path) > SLICE_PATH_LENGTH:  # If path too long
                slice_path.pop(0)  # Remove oldest point

            if len(slice_path) >= 2:  # If enough points for line
                pygame.draw.line(screen, SLICER_COLOR, slice_path[-2], slice_path[-1], SLICER_WIDTH)  # Draw slice line

            if len(slice_path) > 1:  # If multiple points exist
                trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)  # Create trail surface
                for i, point in enumerate(slice_path):  # For each point in path
                    alpha = int(255 * (i / (len(slice_path) - 1))) if len(slice_path) > 1 else 255  # Calculate trail alpha
                    radius = int(TRAIL_BASE_RADIUS * (i / (len(slice_path) - 1))) if len(slice_path) > 1 else TRAIL_BASE_RADIUS  # Calculate trail radius
                    
                    radius = max(1, radius)  # Ensure minimum radius

                    color = TRAIL_COLOR + (alpha,)  # Add alpha to color
                    pygame.draw.circle(trail_surface, color, point, radius)  # Draw trail particle
                screen.blit(trail_surface, (0, 0))  # Draw trail to screen

            if len(slice_path) == SLICE_PATH_LENGTH:  # If path is full length
                start_x, start_y = slice_path[0]  # Get start position
                end_x, end_y = slice_path[-1]  # Get end position
                
                distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)  # Calculate slice distance

                if distance > SLICE_MIN_DISTANCE:  # If slice is long enough
                    slice_detected_in_swipe = False  # Reset slice detection flag
                    for obj in list(fruits_and_bombs):  # For each game object
                        for i in range(len(slice_path) - 1):  # For each path segment
                            p1 = slice_path[i]  # Segment start
                            p2 = slice_path[i+1]  # Segment end
                            if line_rect_intersection(p1, p2, obj.rect):  # If slice intersects object
                                if isinstance(obj, Fruit):  # If object is fruit
                                    if not obj.sliced:  # If not already sliced
                                        if obj.slice():  # Slice the fruit
                                            score += 10  # Add score
                                            if SLICE_SOUND: SLICE_SOUND.play()  # Play slice sound
                                            fruits_and_bombs.remove(obj)  # Remove from active objects
                                            all_sprites.add(obj)  # Keep in all sprites
                                            slice_detected_in_swipe = True  # Mark slice detected
                                            break  # Stop checking this object
                                elif isinstance(obj, Bomb):  # If object is bomb
                                    if not obj.exploded:  # If not already exploded
                                        if obj.explode():  # Explode the bomb
                                            if BOMB_SOUND: BOMB_SOUND.play()  # Play bomb sound
                                            fruits_and_bombs.remove(obj)  # Remove from active objects
                                            all_sprites.add(obj)  # Keep in all sprites
                                            slice_detected_in_swipe = True  # Mark slice detected
                                            if game_state == AppState.GAME_OVER:  # If game ended
                                                break  # Exit inner loop
                        if game_state == AppState.GAME_OVER:  # If game ended
                            break  # Exit outer loop
                            
                    if slice_detected_in_swipe:  # If slice was detected
                        slice_path = []  # Reset slice path
                    
        if game_state != AppState.GAME_OVER:  # If game not over
            all_sprites.update()  # Update all sprites

            for obj in list(fruits_and_bombs):  # For each active object
                if obj.rect.y > HEIGHT + 100:  # If object fell off screen
                    if isinstance(obj, Fruit) and not obj.sliced:  # If unsliced fruit
                        lives -= 1  # Lose a life
                        if lives <= 0:  # If no lives left
                            game_state = AppState.GAME_OVER  # Game over
                    obj.kill()  # Remove sprite
                    fruits_and_bombs.remove(obj)  # Remove from active objects

            if current_time_ms - last_spawn_time > SPAWN_INTERVAL:  # If spawn time reached
                spawn_object()  # Spawn new object
                last_spawn_time = current_time_ms  # Reset spawn timer

            all_sprites.draw(screen)  # Draw all sprites
            draw_text(screen, f"Score: {score}", 40, WIDTH // 2, 30)  # Draw score
            draw_text(screen, f"Lives: {lives}", 40, WIDTH - 100, 30)  # Draw lives
        
        if tracker.is_hand_open() and (current_time_ms - last_open_hand_time > open_hand_debounce_ms):  # If open hand for pause
            game_state = AppState.PAUSED  # Pause game
            last_open_hand_time = current_time_ms  # Update gesture time

    elif game_state == AppState.PAUSED:  # If game paused
        draw_text(screen, "PAUSED", 80, WIDTH // 2, HEIGHT // 2, (255, 255, 0))  # Draw pause text
        draw_text(screen, "Open Palm to Resume", 50, WIDTH // 2, HEIGHT // 2 + 80)  # Draw resume instruction
        
        if tracker.is_hand_open() and (current_time_ms - last_open_hand_time > open_hand_debounce_ms):  # If open hand for resume
            game_state = AppState.PLAYING  # Resume game
            last_open_hand_time = current_time_ms  # Update gesture time

    elif game_state == AppState.GAME_OVER:  # If game over
        draw_text(screen, "GAME OVER", 80, WIDTH // 2, HEIGHT // 2, (255, 0, 0))  # Draw game over text
        draw_text(screen, f"Final Score: {score}", 50, WIDTH // 2, HEIGHT // 2 + 80)  # Draw final score
        draw_text(screen, "Open Palm to Restart", 40, WIDTH // 2, HEIGHT // 2 + 150)  # Draw restart instruction
        
        if tracker.is_hand_open() and (current_time_ms - last_open_hand_time > open_hand_debounce_ms):  # If open hand for restart
            game_state = AppState.PLAYING  # Start new game
            reset_game()  # Reset game state
            last_open_hand_time = current_time_ms  # Update gesture time

    pygame.display.flip()  # Update display
    clock.tick(FPS)  # Maintain framerate

tracker.release()  # Release camera resources
pygame.quit()  # Quit pygame