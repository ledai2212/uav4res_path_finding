import pygame
from Map import Map

# Khởi tạo pygame
pygame.init()

print("Màu đỏ: Obstacle, Màu trắng: Unscanned, màu xám: no interest, number: ấn a để tăng số, s để giảm số, màu xanh: vị trí base, màu xanh lá: vị trí uav_start, ấn q để chuyển đổi giữa các chế độ")

# Nhập số hàng và cột từ terminal
ROWS = int(input("Nhập số lượng hàng: "))
COLS = int(input("Nhập số lượng cột: "))
CELL_SIZE = 20  # Kích thước ô
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE

# #####################################
# num_of_uavs = int(input("Nhập số lượng uavs: "))
# cell_radius = int(input("Nhập cell_radius: "))
# uav_distance = int(input("Nhập uav_distance: "))
# time_charge = int(input("Nhập time_charge: "))
# min_speed = []
# max_speed = []
# for i in range(num_of_uavs):
#     min_speed.append(int(input(f"Nhập min_speed uav thứ {i}: ")))
#     max_speed.append(int(input(f"Nhập max_speed uav thứ {i}: ")))
# dis_threshold = int(input("Nhập dis_threshold: "))
# ######################################


# Màu sắc
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
DARK_GRAY = (80, 80, 80)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)  # Màu xanh biển mới
TEXT_COLOR = (0, 0, 0)  # Màu chữ tương phản với màu trắng

# Tạo cửa sổ
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Grid")

# Font chữ
font = pygame.font.SysFont(None, 20)

# Lưới lưu trạng thái màu và giá trị số của từng ô
grid = [[DARK_GRAY for _ in range(COLS)] for _ in range(ROWS)]
values = [[0 for _ in range(COLS)] for _ in range(ROWS)]
states = [[Map.CellState.NO_INTEREST for _ in range(ROWS)] for _ in range(COLS)]
priority = [[0 for _ in range(ROWS)] for _ in range(COLS)]
selected_green = None  # Ô được chọn trong chế độ xanh lá
selected_blue = None  # Ô được chọn trong chế độ xanh biển

# Chế độ vẽ (0: xám, 1: đỏ, 2: trắng, 3: số, 4: chọn ô xanh lá, 5: chọn ô xanh biển)
draw_mode = 0
current_value = 0  # Giá trị hiện tại để điền vào ô

# Danh sách tên chế độ
mode_names = ["Gray", "Red", "White", "Number", "Select Green", "Select Blue"]

def write_to_input():
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            changer = {
                GRAY: Map.CellState.NO_INTEREST,
                RED: Map.CellState.UNREACHABLE,
                WHITE: Map.CellState.NOT_SCANNED
            }
            states[y][x] = changer[grid[x][y]]
            priority[y][x] = values[x][y]
    
    selected_green_temp = (selected_green[1], selected_green[0])
    selected_blue_temp = (selected_blue[1], selected_blue[0])
    
    f = open("input.py", "w")
    f.write(f"state = {states}\n")
    f.write(f"uav_start = {selected_green_temp}\n")
    f.write(f"base = {selected_blue_temp}\n")
    f.write(f"priority = {priority}\n")
    f.write(f"map_width = {COLS}\n")
    f.write(f"map_height = {ROWS}\n")
    f.write(f"cell_size = {20}\n")
    f.write(f"FPS = {60}\n")
    # f.write(f"cell_radius = {cell_radius}\n")
    # f.write(f"uav_distance = {uav_distance}\n")
    # f.write(f"time_charge = {time_charge}\n")
    # f.write(f"min_speed = {min_speed}\n")
    # f.write(f"max_speed = {max_speed}\n")
    # f.write(f"dis_threshold = {dis_threshold}\n")

    f.close()
    

def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = grid[row][col]
            if selected_green == (row, col):
                color = GREEN
            elif selected_blue == (row, col):
                color = BLUE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)
            
            # Hiển thị số trên ô, kể cả giá trị 0
            text_color = TEXT_COLOR if grid[row][col] == WHITE else (255, 255, 255)
            text_surf = font.render(str(values[row][col]), True, text_color)
            text_rect = text_surf.get_rect(center=(col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2))
            screen.blit(text_surf, text_rect)

def draw_mode_text():
    text = font.render(f"Mode: {mode_names[draw_mode]}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

def handle_mouse_draw(row, col):
    global selected_green, selected_blue
    if draw_mode == 0:
        # Mở rộng vùng xám ra xung quanh
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    grid[nr][nc] = GRAY
                    values[nr][nc] = 0  # Giá trị của ô màu xám luôn là 0
    elif draw_mode == 1:
        grid[row][col] = RED
        values[row][col] = 0  # Giá trị của ô màu đỏ luôn là 0
    elif draw_mode == 2:
        # Mở rộng vùng trắng ra xung quanh
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    grid[nr][nc] = WHITE
                    values[nr][nc] = 1
    elif draw_mode == 3 and grid[row][col] == WHITE:  # Chỉ thay đổi giá trị nếu ô là màu trắng
        
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    values[nr][nc] = current_value
    elif draw_mode == 4:
        selected_green = (row, col)  # Chỉ chọn một ô màu xanh lá
    elif draw_mode == 5:
        selected_blue = (row, col)  # Chỉ chọn một ô màu xanh biển

# Vòng lặp chính
running = True
mouse_pressed = False

while running:
    screen.fill(DARK_GRAY)  # Đặt màu nền là xám đậm
    draw_grid()
    draw_mode_text()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                draw_mode = (draw_mode + 1) % 6  # Chuyển chế độ vẽ
            elif event.key == pygame.K_a and draw_mode == 3:
                current_value = max(0, current_value - 1)  # Giảm giá trị
            elif event.key == pygame.K_s and draw_mode == 3:
                current_value += 1  # Tăng giá trị
            elif event.key == pygame.K_RETURN:
                write_to_input()
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
    
    if mouse_pressed:
        x, y = pygame.mouse.get_pos()
        row, col = y // CELL_SIZE, x // CELL_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            handle_mouse_draw(row, col)
    
    pygame.display.flip()

pygame.quit()
