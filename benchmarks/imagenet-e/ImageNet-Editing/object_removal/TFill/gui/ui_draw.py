from PyQt5 import QtGui, QtCore, QtWidgets


#######################################################################################################
# painter function
#######################################################################################################
class painter(QtWidgets.QWidget):
    """the class for a painter"""
    def __init__(self, parent, image=None):
        super(painter, self).__init__()
        if image is None:
            w = h = 256
        else:
            w, h = image.size().width(), image.size().height()
        self.ParentLink = parent
        self.setPalette(QtGui.QPalette(QtCore.Qt.white))
        self.setAutoFillBackground(True)
        self.setMaximumSize(w, h)
        self.map = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        self.map.fill(QtCore.Qt.black)
        self.image = image
        self.shape = self.ParentLink.shape
        self.CurrentWidth = self.ParentLink.CurrentWidth
        self.MouseLoc = point(0, 0)
        self.LastPos = point(0, 0)
        self.Brush = False
        self.DrawingShapes_free = shapes()
        self.DrawingShapes_rec = shapes()
        self.IsPainting = False
        self.IsEraseing = False
        self.iteration = 0

        self.CurrentColor = colour3(255, 255, 255)

        self.ShapeNum = 0
        self.IsMouseing = False
        self.PaintPanel = 0

    def drawLines(self, painter):
        """draw free-form masks"""
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for i in range(self.DrawingShapes_free.NumberOfShapes()-1):
            T = self.DrawingShapes_free.GetShape(i)
            T1 = self.DrawingShapes_free.GetShape(i + 1)

            if T.ShapeNumber == T1.ShapeNumber:
                pen = QtGui.QPen(QtGui.QColor(T.Color.R, T.Color.G, T.Color.B), T.Width / 2, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(T.Location.X, T.Location.Y, T1.Location.X, T1.Location.Y)

    def drawRectangle(self, painter):
        """draw rectangle mask"""
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for i in range(self.DrawingShapes_rec.NumberOfShapes()-1):
            T = self.DrawingShapes_rec.GetShape(i)
            T1 = self.DrawingShapes_rec.GetShape(i+1)

            if T.ShapeNumber == T1.ShapeNumber:
                pen = QtGui.QPen(QtGui.QColor(T.Color.R, T.Color.G, T.Color.B), T.Width/2, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor(T.Color.R, T.Color.G, T.Color.B))
                painter.drawRects(QtCore.QRect(QtCore.QPoint(T.Location.X, T.Location.Y), QtCore.QPoint(T1.Location.X, T1.Location.Y)))

    def saveDraw(self):
        """save the painted masks"""
        painter = QtGui.QPainter()
        painter.begin(self.map)
        if self.shape == 'line':
            self.drawLines(painter)
        if self.shape == 'rectangle':
            self.drawRectangle(painter)
        painter.end()

    def mousePressEvent(self, event):
        """mouse down event for the drawing"""
        if self.Brush:
            self.IsPainting = True
            self.ShapeNum += 1
            if self.shape == 'rectangle':
                self.DrawingShapes_rec.NewShape(point(event.x(), event.y()), self.CurrentWidth, self.CurrentColor, self.ShapeNum)
            else:
                self.LastPos = point(0, 0)
        else:
            self.IsEraseing = True
        if self.shape == 'rectangle':
            self.DrawingShapes_rec.NewShape(point(event.x(), event.y()), self.CurrentWidth, self.CurrentColor, self.ShapeNum)

    def mouseMoveEvent(self, event):
        """mouse move event to record the track"""
        if self.IsPainting:
            self.MouseLoc = point(event.x(), event.y())
            if self.LastPos.X != self.MouseLoc.X or self.LastPos.Y != self.MouseLoc.Y:
                self.LastPos = point(event.x(), event.y())
                if self.shape == 'line':
                    self.DrawingShapes_free.NewShape(self.LastPos, self.CurrentWidth, self.CurrentColor, self.ShapeNum)
            self.repaint()
        if self.IsEraseing:
            self.MouseLoc = point(event.x(), event.y())
            if self.shape == 'line':
                self.DrawingShapes_free.RemoveShape(self.MouseLoc, 10)
            elif self.shape == 'rectangle':
                self.DrawingShapes_rec.RemoveShape(self.MouseLoc, 10)
            self.repaint()

    def mouseReleaseEvent(self, event):
        """mouse up event"""
        if self.IsEraseing:
            self.IsEraseing = False
            self.repaint()
        elif self.shape == 'rectangle':
            self.DrawingShapes_rec.NewShape(point(event.x(), event.y()), self.CurrentWidth, self.CurrentColor, self.ShapeNum)
            self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        if self.image != None:
            painter.drawImage(0, 0, self.image)
        if self.shape == 'line':
            self.drawLines(painter)
        if self.shape == 'rectangle':
            self.drawRectangle(painter)
        painter.end()
        self.iteration = 0


#######################################################################################################
# base drawing function
#######################################################################################################
class colour3:
    """define the colour plane for the drawing"""
    def __init__(self, nR=0, nG=0, nB=0):
        self.R = nR
        self.G = nG
        self.B = nB


class point():
    """define the location"""
    def __init__(self, nX=0, nY=0):
        self.X = nX
        self.Y = nY

    def Set(self, nX, nY):
        self.X = nX
        self.Y = nY


class shape():
    """define the painter shape"""
    def __init__(self, location=point(0,0), width=1, color=colour3(255, 255, 255), number=0):
        self.Location = location
        self.Width = width
        self.Color = color
        self.ShapeNumber = number


class shapes():
    """a set of shape"""
    def __init__(self):
        self.shapes = []

    def NumberOfShapes(self):
        return len(self.shapes)

    def NewShape(self, location=point(0,0), width=1, color=colour3(255,255,255), number=0):
        Sh = shape(location, width, color, number)
        self.shapes.append(Sh)

    def GetShape(self, Index):
        return self.shapes[Index]

    def RemoveShape(self,  L, threshold):
        i = 0
        while True:
            if (i == len(self.shapes)):
                break
                # Finds if a point is within a certain distance of the point to remove.
            if ((abs(L.X - self.shapes[i].Location.X) < threshold) and (
                    abs(L.Y - self.shapes[i].Location.Y) < threshold)):
                # removes all data for that number
                del self.shapes[i]
                # goes through the rest of the data and adds an extra
                # 1 to defined them as a seprate shape and shuffles on the effect.
                for n in range(len(self.shapes) - i):
                    self.shapes[n + i].ShapeNumber += 1
                # Go back a step so we dont miss a point.
                i -= 1
            i += 1