__author__ = ["Ali Awada", "Melker Forslund"]

# imports
import PyQt6
from PyQt6 import QtGui, QtWidgets
import numpy as np
from PyQt6 import QtCore

# local imports
from parameters import *
from cell import UNBURNED, BURNING, LIT_OUT, BURNED

FUEL_COLOR_SCALE = "gray-yellow-green"
MOISTURE_COLOR_SCALE = "gray-blue"
TEMPERATURE_COLOR_SCALE = "gray-yellow-red"

TEMPERATURE_RANGE = (AMBIENT_TEMPERATURE, MAX_TEMPERATURE)  # Kelvin
FUEL_RANGE = (0.0, 1.0)             # min and max fuel values for visualization
MOISTURE_RANGE = (0.0, 1.0)         # min and max moisture values for visualization

MAX_WIND_SPEED_FOR_PLOT = 40.0  # for color bar scaling in wind visualizer

def get_color(norm_value, color_scale):
    norm_value = min(max(norm_value, 0.0), 1.0)
    if color_scale == "gray-yellow-green":
        # 0 - gray, 0.5 - yellow, 1 - green
        if norm_value < 0.5:
            # Gray to yellow
            t = norm_value / 0.5
            r = int(128 + t * (255 - 128))
            g = int(128 + t * (255 - 128))
            b = int(128 - t * 128)
        else:
            # Yellow to green
            t = (norm_value - 0.5) / 0.5
            r = int(255 - t * 255)
            g = 255
            b = 0
        return QtGui.QColor(r, g, b)
    elif color_scale == "gray-yellow-red":
        # 0 - gray, 0.5 - yellow, 1 - red
        if norm_value < 0.5:
            t = norm_value / 0.5
            r = int(128 + t * (255 - 128))
            g = int(128 + t * (255 - 128))
            b = int(128 - t * 128)
        else:
            t = (norm_value - 0.5) / 0.5
            r = 255
            g = int(255 - t * 255)
            b = 0
        return QtGui.QColor(r, g, b)
    elif color_scale == "gray-blue":
        # 0 - gray, 1 - blue
        r = int(128 * (1 - norm_value))
        g = int(128 * (1 - norm_value))
        b = int(128 + norm_value * (255 - 128))
        return QtGui.QColor(r, g, b)
    else:
        # fallback color
        return QtGui.QColor.fromHsvF(norm_value, 1.0, 1.0)

class ColorBarWidget(QtWidgets.QWidget):
    def __init__(self, height, width=30, min_value=0.0, max_value=1.0, color_scale="gray-yellow-green", parent=None):
        super().__init__(parent)
        self.height = height
        self.width = width
        self.min_value = min_value
        self.max_value = max_value
        self.color_scale = color_scale
        self.setFixedSize(self.width, self.height)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        for y in range(self.height):
            value = self.min_value + (self.max_value - self.min_value) * (self.height - y - 1) / self.height
            norm_value = (value - self.min_value) / (self.max_value - self.min_value) if self.max_value != self.min_value else 0.0
            color = get_color(norm_value, self.color_scale)
            painter.fillRect(0, y, self.width, 1, color)

        painter.setPen(QtGui.QColor("black"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        ticks = np.linspace(self.min_value, self.max_value, 5)

        for tick in ticks:
            fraction = (tick - self.min_value) / (self.max_value - self.min_value) if self.max_value != self.min_value else 0.0
            y = self.height - int(fraction * self.height)
            painter.drawLine(self.width - 8, y, self.width, y)
            painter.drawText(2, y + 4, f"{tick:.2f}")

        painter.end()

class GridWidget(QtWidgets.QWidget):
    def __init__(self, data, cell_size=5, min_value=0.0, max_value=1.0, color_scale="gray-yellow-green", parent=None):
        super().__init__(parent)
        self.data = data
        self.size = data.shape[0]
        self.cell_size = cell_size
        self.min_value = min_value
        self.max_value = max_value
        self.color_scale = color_scale
        self.setFixedSize(self.size * self.cell_size, self.size * self.cell_size)

    def set_grid(self, new_data):
        self.data = new_data
        self.repaint()

    def set_range(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.repaint()

    def set_color_scale(self, color_scale):
        self.color_scale = color_scale
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        for col in range(self.size):
            for row in range(self.size):
                value = self.data[col, row]
                norm_value = (value - self.min_value) / (self.max_value - self.min_value) if self.max_value != self.min_value else 0.0
                norm_value = min(max(norm_value, 0.0), 1.0)
                color = get_color(norm_value, self.color_scale)
                painter.fillRect(row * self.cell_size, col * self.cell_size, self.cell_size, self.cell_size, color)
        painter.end()

class WindGridWidget(QtWidgets.QWidget):
    def __init__(self, wind_u, wind_v, cell_size=9, max_speed_range=50.0, parent=None):
        super().__init__(parent)
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.size = wind_u.shape[0]
        self.cell_size = cell_size
        self.max_speed_range = max_speed_range
        self.setFixedSize(self.size * self.cell_size, self.size * self.cell_size)

    def set_wind(self, wind_u, wind_v):
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Calculate wind speeds for color mapping
        wind_speed = np.sqrt(self.wind_u**2 + self.wind_v**2)
        max_speed = self.max_speed_range
        
        for i in range(self.size):
            for j in range(self.size):
                u = self.wind_u[i, j]
                v = self.wind_v[i, j]
                speed = wind_speed[i, j]
                
                # Normalize speed for color
                normalized_speed = speed / max_speed

                color = get_color(normalized_speed, "gray-yellow-red")
                
                # get center of cell
                cx = j * self.cell_size + self.cell_size // 2
                cy = i * self.cell_size + self.cell_size // 2
                
                # Arrow length proportional to speed
                arrow_scale = 0.3 * self.cell_size
                if speed > 0:
                    dx = (u / speed) * arrow_scale
                    dy = (v / speed) * arrow_scale
                else:
                    dx = 0
                    dy = 0
                
                # Draw arrow
                painter.setPen(QtGui.QPen(color, 1.5))
                painter.setBrush(QtGui.QBrush(color))
                
                # Arrow line
                painter.drawLine(int(cx), int(cy), int(cx + dx), int(cy + dy))
                
                # Arrow head
                if speed > 0.1:
                    arrow_size = 2.5
                    angle = np.arctan2(dy, dx)
                    
                    # Arrow tip point
                    tip_x = cx + dx
                    tip_y = cy + dy
                    
                    # Arrow head points
                    left_angle = angle + 2.8
                    right_angle = angle - 2.8
                    
                    left_x = tip_x - arrow_size * np.cos(left_angle)
                    left_y = tip_y - arrow_size * np.sin(left_angle)
                    right_x = tip_x - arrow_size * np.cos(right_angle)
                    right_y = tip_y - arrow_size * np.sin(right_angle)
                    
                    polygon = QtGui.QPolygonF([
                                                QtCore.QPointF(tip_x, tip_y),
                                                QtCore.QPointF(left_x, left_y),
                                                QtCore.QPointF(right_x, right_y)
                                              ])
                    painter.drawPolygon(polygon)
        
        painter.end()

class StateGridWidget(QtWidgets.QWidget):
    def __init__(self, state_data, cell_size=9, parent=None):
        super().__init__(parent)
        self.state_data = state_data
        self.size = state_data.shape[0]
        self.cell_size = cell_size
        self.setFixedSize(self.size * self.cell_size, self.size * self.cell_size)
        
        # Define colors for each state
        self.state_colors = {
            UNBURNED: QtGui.QColor(34, 139, 34),    # Green
            BURNING: QtGui.QColor(255, 0, 0),       # Red
            LIT_OUT: QtGui.QColor(255, 140, 0),     # Orange
            BURNED: QtGui.QColor(0, 0, 0)           # Black
        }

    def set_state(self, state_data):
        self.state_data = state_data
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        
        for col in range(self.size):
            for row in range(self.size):
                state = int(self.state_data[col, row])
                color = self.state_colors.get(state, QtGui.QColor(128, 128, 128))
                painter.fillRect(row * self.cell_size, col * self.cell_size, 
                               self.cell_size, self.cell_size, color)
        
        painter.end()

class StateLegendWidget(QtWidgets.QWidget):
    def __init__(self, height, width=120, parent=None):
        super().__init__(parent)
        self.height = height
        self.width = width
        self.setFixedSize(self.width, self.height)
        
        self.state_colors = {
                            "Unburned": QtGui.QColor(34, 139, 34),
                            "Burning": QtGui.QColor(255, 0, 0),
                            "Lit Out": QtGui.QColor(255, 140, 0),
                            "Burned": QtGui.QColor(0, 0, 0)
                            }

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor("black"))
        
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        y_offset = 20
        spacing = self.height // 5
        
        for label, color in self.state_colors.items():
            # Draw colored box
            painter.fillRect(10, y_offset, 20, 20, color)
            painter.drawRect(10, y_offset, 20, 20)
            
            # Draw label
            painter.drawText(35, y_offset + 15, label)
            y_offset += spacing
        
        painter.end()

class StateGridWithLegend(QtWidgets.QWidget):
    def __init__(self, state_data, parent=None):
        super().__init__(parent)
        self.state_widget = StateGridWidget(state_data, cell_size=9)
        self.legend = StateLegendWidget(height=self.state_widget.size * self.state_widget.cell_size)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.state_widget)
        layout.addWidget(self.legend)
        self.setLayout(layout)

    def set_state(self, state_data):
        self.state_widget.set_state(state_data)

class GridWithColorBar(QtWidgets.QWidget):
    def __init__(self, data, min_value=0.0, max_value=1.0, color_scale="gray-yellow-green", parent=None):
        super().__init__(parent)
        self.grid_widget = GridWidget(data, min_value=min_value, max_value=max_value, color_scale=color_scale)
        self.color_bar = ColorBarWidget(height=self.grid_widget.size * self.grid_widget.cell_size, min_value=min_value, max_value=max_value, color_scale=color_scale)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid_widget)
        layout.addWidget(self.color_bar)
        self.setLayout(layout)

    def set_grid(self, new_data):
        self.grid_widget.set_grid(new_data)

    def set_range(self, min_value, max_value):
        self.grid_widget.set_range(min_value, max_value)
        self.color_bar.min_value = min_value
        self.color_bar.max_value = max_value
        self.color_bar.repaint()

    def set_color_scale(self, color_scale):
        self.grid_widget.set_color_scale(color_scale)
        self.color_bar.color_scale = color_scale
        self.color_bar.repaint()

class WindGridWithColorBar(QtWidgets.QWidget):
    def __init__(self, wind_u, wind_v, max_speed_range=50.0, parent=None):
        super().__init__(parent)
        self.wind_widget = WindGridWidget(wind_u, wind_v, max_speed_range=max_speed_range)
        # Use static max speed for color bar
        self.color_bar = ColorBarWidget(height=self.wind_widget.size * self.wind_widget.cell_size, 
                                        min_value=0.0, max_value=max_speed_range, color_scale="gray-yellow-red")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.wind_widget)
        layout.addWidget(self.color_bar)
        self.setLayout(layout)

    def set_wind(self, wind_u, wind_v):
        self.wind_widget.set_wind(wind_u, wind_v)

class StateVisualizer(QtWidgets.QWidget):
    def __init__(self, world, parent=None):
        super().__init__(parent)
        self.world = world
        state_data = world.get_state_array()

        self.state_widget = StateGridWithLegend(state_data)

        # Label
        state_label = QtWidgets.QLabel("Cell states")
        state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = state_label.font()
        font.setPointSize(14)
        font.setBold(True)
        state_label.setFont(font)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(state_label)
        main_layout.addWidget(self.state_widget)
        self.setLayout(main_layout)
        self.setWindowTitle("State visualization")

    def update_visualization(self):
        state_data = self.world.get_state_array()
        self.state_widget.set_state(state_data)

class WindVisualizer(QtWidgets.QWidget):
    def __init__(self, world, max_speed_range=100.0, speed_mean=0.0, direction_degrees=0.0, parent=None):
        super().__init__(parent)
        self.world = world
        self.direction_degrees = direction_degrees
        wind_u, wind_v = world.get_wind_arrays()
        
        self.wind_widget = WindGridWithColorBar(wind_u, wind_v, max_speed_range=max_speed_range)
        
        # Create custom label with arrow
        self.wind_label = WindDirectionLabel(speed_mean, direction_degrees, max_speed_range=max_speed_range)
        
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.wind_label)
        main_layout.addWidget(self.wind_widget)
        self.setLayout(main_layout)
        self.setWindowTitle("Wind visualization")
    
    def update_visualization(self):
        wind_u, wind_v = self.world.get_wind_arrays()
        self.wind_widget.set_wind(wind_u, wind_v)

class WindDirectionLabel(QtWidgets.QWidget):
    def __init__(self, speed_mean, direction_degrees, max_speed_range=50.0, parent=None):
        super().__init__(parent)
        self.speed_mean = speed_mean
        self.direction_degrees = direction_degrees
        self.max_speed_range = max_speed_range
        self.setFixedHeight(60)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Draw text
        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        
        text = f"Wind Field - Speed: {self.speed_mean:.1f} m/s - Direction: {self.direction_degrees:.0f} degrees"
        text_rect = painter.fontMetrics().boundingRect(text)
        text_x = (self.width() - text_rect.width()) // 2 - 30
        text_y = self.height() // 2 + text_rect.height() // 3
        
        painter.setPen(QtGui.QColor("black"))
        painter.drawText(text_x, text_y, text)
        
        # Draw arrow to the right of text
        arrow_x = text_x + text_rect.width() + 15
        arrow_y = self.height() // 2
        
        angle_radians = np.deg2rad(self.direction_degrees)
        dx = 20 * np.sin(angle_radians) 
        dy = -20 * np.cos(angle_radians)  
        
        # Calculate arrow color based on speed
        normalized_speed = self.speed_mean / self.max_speed_range
        arrow_color = get_color(normalized_speed, "gray-yellow-red")
        
        # Draw arrow shaft
        painter.setPen(QtGui.QPen(arrow_color, 3))
        painter.drawLine(int(arrow_x), int(arrow_y), int(arrow_x + dx), int(arrow_y + dy))
        
        # Draw arrow head
        painter.setBrush(QtGui.QBrush(arrow_color))
        
        tip_x = arrow_x + dx
        tip_y = arrow_y + dy
        
        arrow_head_size = 6
        left_angle = angle_radians + 2.8
        right_angle = angle_radians - 2.8
        
        left_x = tip_x - arrow_head_size * np.sin(left_angle)
        left_y = tip_y + arrow_head_size * np.cos(left_angle)
        right_x = tip_x - arrow_head_size * np.sin(right_angle)
        right_y = tip_y + arrow_head_size * np.cos(right_angle)
        
        polygon = QtGui.QPolygonF([
                                    QtCore.QPointF(tip_x, tip_y),
                                    QtCore.QPointF(left_x, left_y),
                                    QtCore.QPointF(right_x, right_y)
                                 ])
        painter.drawPolygon(polygon)
        
        painter.end()

class WorldVisualizer(QtWidgets.QWidget):
    def __init__(self, world, parent=None):
        super().__init__(parent)
        self.world = world

        fuel_grid = world.get_fuel_array()
        moisture_grid = world.get_moisture_array()
        temp_grid = world.get_temperature_array()

        self.fuel_widget = GridWithColorBar(fuel_grid, min_value=FUEL_RANGE[0], max_value=FUEL_RANGE[1], color_scale=FUEL_COLOR_SCALE)
        self.moisture_widget = GridWithColorBar(moisture_grid, min_value=MOISTURE_RANGE[0], max_value=MOISTURE_RANGE[1], color_scale=MOISTURE_COLOR_SCALE)
        self.temp_widget = GridWithColorBar(temp_grid, min_value=TEMPERATURE_RANGE[0], max_value=TEMPERATURE_RANGE[1], color_scale=TEMPERATURE_COLOR_SCALE)

        # Labels
        fuel_label = QtWidgets.QLabel("Fuel")
        moisture_label = QtWidgets.QLabel("Moisture")
        temp_label = QtWidgets.QLabel("Temperature")
        for lbl in [fuel_label, moisture_label, temp_label]:
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            font = lbl.font()
            font.setPointSize(12)
            font.setBold(True)
            lbl.setFont(font)

        # Layout
        grid_layout = QtWidgets.QHBoxLayout()
        grid_layout.setSpacing(30)
        grid_layout.addWidget(self.fuel_widget)
        grid_layout.addWidget(self.moisture_widget)
        grid_layout.addWidget(self.temp_widget)

        label_layout = QtWidgets.QHBoxLayout()
        label_layout.setSpacing(30)
        label_layout.addWidget(fuel_label)
        label_layout.addWidget(moisture_label)
        label_layout.addWidget(temp_label)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addLayout(label_layout)
        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)

    def update_visualization(self):
        
        fuel_grid = self.world.get_fuel_array()
        moisture_grid = self.world.get_moisture_array()
        temp_grid = self.world.get_temperature_array()
        
        self.fuel_widget.set_grid(fuel_grid)
        self.moisture_widget.set_grid(moisture_grid)
        self.temp_widget.set_grid(temp_grid)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    from create_world import load_world  
    world = load_world(WORLD_FILE_NAME)
    visualizer = WorldVisualizer(world)
    visualizer.show()

    app.exec()
