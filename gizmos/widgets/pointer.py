from PySide2 import QtCore, QtGui, QtWidgets

from magicroto.gizmos.core.singleton import SingletonByNode
from magicroto.gizmos.widgets.signal import Signal
from magicroto.utils import image_utils, common_utils
from magicroto.utils.logger import logger


class Pointer(QtWidgets.QWidget, SingletonByNode):
    MAX_WIDTH = 1080  # Set your maximum width limit here

    def __init__(self, node, initial_points=None, initial_labels=None, easyRoto=None):
        super(Pointer, self).__init__()
        self.coordinatesChanged = Signal(name='points')  # SignalEmitter()  # Moved it inside the class
        # self.coordinatesChanged = QtCore.Signal(list)
        self.easyRoto = easyRoto
        self.node = node
        self._instances[node] = self
        self.original_image = None
        self.image = None  # Variable to store the image
        self.scaled_image = None
        self.points = [] if initial_points is None else initial_points # Initialize with given points or an empty list
        self.points = [tuple(p) for p in self.points]
        self.labels = [] if initial_labels is None else initial_labels  # Initialize with given labels or an empty list
        self.data = {'point_coords': self.points, 'point_labels': self.labels}
        logger.debug(f'in Pointer widget {self.points}, {self.labels}')
        self._canvas_width = self.MAX_WIDTH  # Default values
        self._canvas_height = self.MAX_WIDTH  # Default values

        self.original_width = None  # Store the original dimensions of the image
        self.original_height = None

        self._scale = 1.0
        self.max_scale = 2.0
        self.min_scale = .5

        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )

    @property
    def scale(self):
        return min(max(self._scale, self.min_scale), self.max_scale)

    @scale.setter
    def scale(self, scale):
        self._scale = min(max(scale, self.min_scale), self.max_scale)

    def overlay_masks(self, masks):
        self.image = image_utils.overlay_boolean_arrays_on_image(self.original_image, masks)
        self.scaled_image = self.image.scaled(self.scaled_image.width(), self.scaled_image.height(),
                                              QtCore.Qt.KeepAspectRatio)
        self.update()

    @property
    def canvas_width(self):
        return int(self._canvas_width)

    @canvas_width.setter
    def canvas_width(self, width):
        if self.original_width:
            width = max(min(width, self.original_width * 2), self.original_width * .2)
        self._canvas_width = width

    @property
    def canvas_height(self):
        return int(self._canvas_height)

    @canvas_height.setter
    def canvas_height(self, height):
        if self.original_height:
            height = max(min(height, self.original_height * 2), self.original_height * .2)
        self._canvas_height = height

    def connect_signal(self, slot):
        self.coordinatesChanged.connect(slot)

    def disconnect_signal(self, slot):
        self.coordinatesChanged.disconnect(slot)

    def paintEvent(self, e):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(painter.Antialiasing)

        # Draw the image if it exists
        if self.scaled_image:
            painter.drawImage(0, 0, self.scaled_image)

        for point, label in zip(self.points, self.labels):
            x, y = point
            if label:
                painter.setPen(QtGui.QPen(QtCore.Qt.green, 5))  # For example, a green dot
            else:
                painter.setPen(QtGui.QPen(QtCore.Qt.red, 5))  # For example, a red dot
            painter.drawPoint(x * self.scale, y * self.scale)  # Draw scaled points

    def makeUI(self):
        return self

    def sizeHint(self):
        return QtCore.QSize(self.canvas_width, self.canvas_height)

    def mousePressEvent(self, e):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        x, y = e.x(), e.y()
        scaled_x = int(x / self.scale)
        scaled_y = int(y / self.scale)

        if modifiers == QtCore.Qt.ControlModifier:  # Ctrl+click
            # Find and remove the point if it's close enough to click location
            threshold = 10  # Choose a suitable threshold
            for point in self.points:
                px, py = point
                if abs(px - scaled_x) <= threshold and abs(py - scaled_y) <= threshold:
                    index = self.points.index(point)
                    self.labels.pop(index)
                    self.points.pop(index)
                    break
                
        elif self.scaled_image and 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
            self.points.append((scaled_x, scaled_y))
            label = 0 if modifiers == QtCore.Qt.AltModifier else 1
            self.labels.append(label)

        self.data.update({'point_coords': self.points, 'point_labels': self.labels})

        self.coordinatesChanged.emit(self.data)  # Emit the entire list
        self.update()

    def updateCanvasSize(self):

        width = int(self.original_width * self.scale)
        height = int(self.original_height * self.scale)

        self.scaled_image = self.image.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        self.canvas_width = width
        self.canvas_height = height
        self.setFixedSize(width, height)
        self.update()

    def clear(self):
        self.image = self.original_image.copy()
        self.scaled_image = self.image.scaled(self.scaled_image.width(), self.scaled_image.height(),
                                              QtCore.Qt.KeepAspectRatio)
        self.points = []
        self.labels = []
        self.update()

    def load_image(self, filepath):
        """Load an image from the specified file path and set canvas size based on image dimensions."""
        self.image = QtGui.QImage(filepath)
        self.original_image = self.image.copy()
        # self.points = []
        if not self.image.isNull():
            self.original_width, self.original_height = self.image.width(), self.image.height()
            self.scale = min(self.MAX_WIDTH / self.original_width, 1)
            self.updateCanvasSize()

            parent = self.parent()
            _parent = parent
            while _parent:
                _parent = parent.parent()
                if _parent:
                    parent = _parent

            if parent is not None and hasattr(parent, 'setGeometry'):
                self.setGeometry(100, 100, self.canvas_width * 2, self.canvas_height)

        self.update()  # Request a repaint to show the loaded image

        if self.easyRoto is not None and hasattr(self.easyRoto, 'load_image'):
            self.easyRoto.load_image(filepath)

            if all([self.points, self.labels]):
                self.coordinatesChanged.emit(self.data)


class ZoomablePointer(QtWidgets.QWidget):
    def __init__(self, node, initial_points=None, initial_labels=None, easyRoto=None):
        super(ZoomablePointer, self).__init__()

        self.pointer = Pointer(node, initial_points, initial_labels, easyRoto=easyRoto)

        self.scrollArea = QtWidgets.QScrollArea()  # Create a QScrollArea
        self.scrollArea.setWidget(self.pointer)  # Set the Pointer widget inside the QScrollArea
        self.scrollArea.setWidgetResizable(True)  # Allow the widget to resize within the scroll area

        self.layout = QtWidgets.QHBoxLayout()

        self.buttonLayout = QtWidgets.QVBoxLayout()
        self.buttonLayout.setAlignment(QtCore.Qt.AlignTop)

        self.zoomInButton = QtWidgets.QPushButton("+")
        self.zoomInButton.setFixedSize(30, 30)

        self.zoomOutButton = QtWidgets.QPushButton("-")
        self.zoomOutButton.setFixedSize(30, 30)

        style = self.style()
        icon = style.standardIcon(QtWidgets.QStyle.SP_TrashIcon)  # or QStyle.SP_DialogResetButton
        self.clearButton = QtWidgets.QPushButton()
        self.clearButton.setIcon(icon)
        self.clearButton.setFixedSize(30, 30)

        self.zoomInButton.clicked.connect(self.zoom_in)
        self.zoomOutButton.clicked.connect(self.zoom_out)
        self.clearButton.clicked.connect(self.clear)

        self.buttonLayout.addWidget(self.zoomInButton)
        self.buttonLayout.addWidget(self.zoomOutButton)
        self.buttonLayout.addWidget(self.clearButton)

        self.layout.addLayout(self.buttonLayout)
        self.layout.addWidget(self.scrollArea)  # Add the scroll area to the layout

        self.setLayout(self.layout)

    def clear(self):
        self.pointer.clear()

    def zoom_in(self):
        scale = (self.pointer.canvas_width + 50) / self.pointer.original_width
        self.pointer.scale = scale
        self.pointer.updateCanvasSize()

    def zoom_out(self):
        scale = (self.pointer.canvas_width - 50) / self.pointer.original_width
        self.pointer.scale = scale
        self.pointer.updateCanvasSize()

    # Forward to connect and disconnect methods to the embedded Pointer instance
    def connect_signal(self, slot):
        self.pointer.connect_signal(slot)

    def disconnect_signal(self, slot):
        self.pointer.disconnect_signal(slot)

    def __getattr__(self, name):
        # Forward the attribute access to self.pointer if the attribute
        # doesn't exist in the current class (ZoomablePointer).
        return getattr(self.pointer, name)

# if __name__ == '1__main__':
#     # Create a NoOp node on which we'll add the knobs
#     node = nuke.createNode("NoOp", inpanel=False)
#     img = r'C:/Users/mellithy/nuke-stable-diffusion/SD_Txt2Img/mdjrny-v4.ckpt/20230911_072213/20230911_072213_1_1.png'
#     nImg = r'C:\Users\mellithy\nuke-stable-diffusion\SD_Txt2Img\cyberrealistic_v31\20230901_140229\20230901_140229_1_1.png'
#     img = r'C:/Users/mellithy/nuke-stable-diffusion/SD_Txt2Img/mdjrny-v4.ckpt/20230911_072213/20230911_072213_1_1.png'
#     # Custom knob
#     knob = nuke.PyCustom_Knob("EasyRotoKnob", "", "Pointer(nuke.thisNode())")
#     node.addKnob(knob)
#     node.showControlPanel()
#     easyRoto = Pointer.instance(node)
#     easyRoto.loadImage(nImg)
