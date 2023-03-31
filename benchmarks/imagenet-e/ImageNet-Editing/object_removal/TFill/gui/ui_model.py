from gui.ui_win import Ui_Form
from gui.ui_draw import *
from PIL import Image, ImageQt
import numpy as np
import random, io, os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from util import task, util
from dataloader.image_folder import make_dataset
from dataloader.data_loader import get_transform
from model import create_model


class ui_model(QtWidgets.QWidget, Ui_Form):
    """define the class of UI"""
    shape = 'line'
    CurrentWidth = 1

    def __init__(self, opt):
        super(ui_model, self).__init__()

        self.setupUi(self)

        self.opt = opt
        self.show_result_flag = False
        self.mask_type = None
        self.img_power = None
        self.model_names = ['celeba', 'ffhq', 'imagenet', 'places2']
        self.img_root = './examples/'
        self.img_files = ['celeba/img', 'ffhq/img', 'imagenet/img', 'places2/img']

        self.show_logo()

        self.comboBox.activated.connect(self.load_model)        # select model
        self.pushButton_2.clicked.connect(self.select_image)      # manually select an image
        self.pushButton_3.clicked.connect(self.random_image)    # randomly select an image
        self.pushButton_4.clicked.connect(self.load_mask)       # manually select a mask
        self.pushButton_5.clicked.connect(self.random_mask)     # randomly select a mask

        # draw/erasure the mask
        self.radioButton.toggled.connect(lambda: self.draw_mask('line'))          # draw the line
        self.radioButton_2.toggled.connect(lambda: self.draw_mask('rectangle'))   # draw the rectangle
        self.radioButton_3.toggled.connect(lambda: self.draw_mask('center'))      # center mask
        self.spinBox.valueChanged.connect(self.change_thickness)
        self.pushButton.clicked.connect(self.clear_mask)

        # fill image
        self.pushButton_6.clicked.connect(self.fill_image)
        self.comboBox_2.activated.connect(self.show_result)
        self.pushButton_7.clicked.connect(self.save_result)

        opt.preprocess = 'scale_shortside'
        self.transform_o = get_transform(opt, convert=False, augment=False)
        self.pil2tensor = transforms.ToTensor()

    def show_logo(self):
        """Show the logo of NTU and BTC"""
        img = QtWidgets.QLabel(self)
        img.setGeometry(1000, 10, 140, 50)

        pixmap = QtGui.QPixmap("./gui/logo/NTU_logo.jpg")  # read examples
        pixmap = pixmap.scaled(140, 140, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        img.setPixmap(pixmap)
        img.show()
        img1 = QtWidgets.QLabel(self)
        img1.setGeometry(1200, 10, 70, 50)

        pixmap1 = QtGui.QPixmap("./gui/logo/BTC_logo.png")  # read examples
        pixmap1 = pixmap1.scaled(70, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        img1.setPixmap(pixmap1)
        img1.show()

    def show_image(self, img):
        """Show the masked examples"""
        show_img = img.copy()
        if self.mask_type == 'center':
            sub_img = Image.fromarray(np.uint8(255 * np.ones((int(self.pw/2), int(self.pw/2), 3))))
            mask = Image.fromarray(np.uint8(255 * np.ones((int(self.pw/2), int(self.pw/2)))))
            show_img.paste(sub_img, box=(int(self.pw/4), int(self.pw/4)), mask=mask)
        elif self.mask_type == 'external':
            mask = Image.open(self.mname).resize(self.img_power.size).convert('RGB')
            mask_L = Image.open(self.mname).resize(self.img_power.size).convert('L')
            show_img = Image.composite(mask, show_img, mask_L)
        self.new_painter(ImageQt.ImageQt(show_img))

    def show_result(self):
        """Show different kind examples"""
        value = self.comboBox_2.currentIndex()
        if value == 0:
            self.new_painter(ImageQt.ImageQt(self.img_power))
        elif value == 1:
            masked_img = torch.where(self.mask > 0, self.img_m, torch.ones_like(self.img_m))
            masked_img = Image.fromarray(util.tensor2im(masked_img.detach()))
            self.new_painter(ImageQt.ImageQt(masked_img))
        elif value == 2:
            if 'refine' in self.opt.coarse_or_refine:
                img_out = Image.fromarray(util.tensor2im(self.img_ref_out.detach()))
            else:
                img_out = Image.fromarray(util.tensor2im(self.img_out.detach()))
            self.new_painter(ImageQt.ImageQt(img_out))

    def save_result(self):
        """Save the results to the disk"""
        util.mkdir(self.opt.results_dir)
        img_name = self.fname.split('/')[-1]
        data_name = self.opt.img_file.split('/')[-1].split('.')[0]

        original_name = '%s_%s_%s' % ('original', data_name, img_name)  # save the original image
        original_path = os.path.join(self.opt.results_dir, original_name)
        img_original = util.tensor2im(self.img_truth)
        util.save_image(img_original, original_path)

        mask_name = '%s_%s_%d_%s' % ('mask', data_name, self.PaintPanel.iteration, img_name)
        mask_path = os.path.join(self.opt.results_dir, mask_name)
        mask = self.mask.repeat(1, 3, 1, 1)
        img_mask = util.tensor2im(1-mask)
        util.save_image(img_mask, mask_path)

        #save masked image
        masked_img_name = '%s_%s_%d_%s' % ('masked_img', data_name, self.PaintPanel.iteration, img_name)
        img_path = os.path.join(self.opt.results_dir, masked_img_name)
        img = torch.where(self.mask < 0.2, torch.ones_like(self.img_truth), self.img_truth)
        masked_img = util.tensor2im(img)
        util.save_image(masked_img, img_path)

        # save the generated results
        img_g_name = '%s_%s_%d_%s' % ('g', data_name, self.PaintPanel.iteration, img_name)
        img_path = os.path.join(self.opt.results_dir, img_g_name)
        img_g = util.tensor2im(self.img_g)
        util.save_image(img_g, img_path)

        # save the results
        result_name = '%s_%s_%d_%s' % ('out', data_name, self.PaintPanel.iteration, img_name)
        result_path = os.path.join(self.opt.results_dir, result_name)
        img_result = util.tensor2im(self.img_out)
        util.save_image(img_result, result_path)

        # save the refined results
        if 'tc' in self.opt.model and 'refine' in self.opt.coarse_or_refine:
            result_name = '%s_%s_%d_%s' % ('ref', data_name, self.PaintPanel.iteration, img_name)
            result_path = os.path.join(self.opt.results_dir, result_name)
            img_result = util.tensor2im(self.img_ref_out)
            util.save_image(img_result, result_path)

    def load_model(self):
        """Load different kind models"""
        value = self.comboBox.currentIndex()
        if value == 0:
            raise NotImplementedError("Please choose a model")
        else:
            index = value-1    # define the model type and dataset type
            self.opt.name = self.model_names[index]
            self.opt.img_file = self.img_root + self.img_files[index % len(self.img_files)]
        self.model = create_model(self.opt)
        self.model.setup(self.opt)

    def load_image(self, fname):
        """Load the image"""
        self.img_o = Image.open(fname).convert('RGB')
        self.ow, self.oh = self.img_o.size
        self.img_power = self.transform_o(self.img_o)
        self.pw, self.ph = self.img_power.size

        return self.img_power

    def select_image(self):
        """Load the image"""
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', self.opt.img_file, '*')
        img = self.load_image(self.fname)

        self.mask_type = 'none'
        self.show_image(img)

    def random_image(self):
        """Random load the test image"""
        image_paths, image_size = make_dataset(self.opt.img_file)
        item = random.randint(0, image_size-1)
        self.fname = image_paths[item]
        img = self.load_image(self.fname)

        self.mask_type = 'none'
        self.show_image(img)

    def load_mask(self):
        """Load a mask"""
        self.mask_type = 'external'
        self.mname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the mask', self.opt.mask_file,'*')

        self.show_image(self.img_power)

    def random_mask(self):
        """Random load the test mask"""
        if self.opt.mask_file == 'none':
            raise NotImplementedError("Please input the mask path")
        self.mask_type = 'external'
        mask_paths, mask_size = make_dataset(self.opt.mask_file)
        item = random.randint(0, mask_size - 1)
        self.mname = mask_paths[item]

        self.show_image(self.img_power)

    def read_mask(self):
        """Read the mask from the painted plain"""
        self.PaintPanel.saveDraw()
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QBuffer.ReadWrite)
        self.PaintPanel.map.save(buffer, 'PNG')
        pil_im = Image.open(io.BytesIO(buffer.data()))

        return pil_im

    def new_painter(self, image=None):
        """Build a painter to load and process the image"""
        # painter
        self.PaintPanel = painter(self, image)
        self.PaintPanel.close()
        if image is not None:
            w, h = image.size().width(), image.size().height()
            self.stackedWidget.setGeometry(QtCore.QRect(250+int(512-w/2), 100+int(128-h/8), w, h))
        self.stackedWidget.insertWidget(0, self.PaintPanel)
        self.stackedWidget.setCurrentWidget(self.PaintPanel)

    def change_thickness(self, num):
        """Change the width of the painter"""
        self.CurrentWidth = num
        self.PaintPanel.CurrentWidth = num

    def draw_mask(self, masktype):
        """Draw the mask"""
        if masktype == 'center':
            self.mask_type = 'center'
            if self.img_power is not None:
                self.show_image(self.img_power)
        else:
            self.mask_type = 'draw'
            self.shape = masktype
            self.PaintPanel.shape = masktype

    def clear_mask(self):
        """Clear the mask"""
        self.mask_type = 'draw'
        if self.PaintPanel.Brush:
            self.PaintPanel.Brush = False
        else:
            self.PaintPanel.Brush = True

    def set_input(self):
        """Set the input for the network"""
        img_o = self.pil2tensor(self.img_o).unsqueeze(0)
        img = self.pil2tensor(self.img_power).unsqueeze(0)
        if self.mask_type == 'draw':
            # get the test mask from painter
            mask = self.read_mask()
            mask = torch.autograd.Variable(self.pil2tensor(mask)).unsqueeze(0)[:, 0:1, :, :]
        elif self.mask_type == 'center':
            mask = torch.zeros_like(img)[:, 0:1, :, :]
            mask[:, :, int(self.pw/4):int(3*self.pw/4), int(self.ph/4):int(3*self.ph/4)] = 1
        elif self.mask_type == 'external':
            mask = self.pil2tensor(Image.open(self.mname).resize((self.pw, self.ph)).convert('L')).unsqueeze(0)
        mask = (mask < 0.5).float()
        if len(self.opt.gpu_ids) > 0:
            img = img.cuda(self.opt.gpu_ids[0])
            mask = mask.cuda(self.opt.gpu_ids[0])
            img_o = img_o.cuda(self.opt.gpu_ids[0])

        self.mask = mask
        self.img_org = img_o * 2 - 1
        self.img_truth = img * 2 - 1
        self.img_m = self.mask * self.img_truth

    def fill_image(self):
        """Forward to get the completed results"""
        self.set_input()
        if self.PaintPanel.iteration < 1:
                with torch.no_grad():
                    fixed_img = F.interpolate(self.img_m, size=[self.opt.fixed_size, self.opt.fixed_size], mode='bicubic', align_corners=True).clamp(-1, 1)
                    fixed_mask = (F.interpolate(self.mask, size=[self.opt.fixed_size, self.opt.fixed_size], mode='bicubic', align_corners=True) > 0.9).type_as(fixed_img)
                    out, mask = self.model.netE(fixed_img, mask=fixed_mask, return_mask=True)
                    out = self.model.netT(out, mask, bool_mask=False)
                    self.img_g = self.model.netG(out)
                    img_g_org = F.interpolate(self.img_g, size=self.img_truth.size()[2:], mode='bicubic', align_corners=True).clamp(-1, 1)
                    self.img_out = self.mask * self.img_truth + (1 - self.mask) * img_g_org
                    if 'refine' in self.opt.coarse_or_refine:
                        img_ref = self.model.netG_Ref(self.img_out, mask=self.mask)
                        self.img_ref_out = self.mask * self.img_truth + (1 - self.mask) * img_ref
                    print('finish the completion')

        self.show_result_flag = True
        self.show_result()