# image_viewer_slideshow.py

import glob
import os
from os.path import dirname

import wx
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
import imagehash
from collections import namedtuple
#https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl

def two_image_is_identical(file_1, file_2):
    hash1 = imagehash.phash(Image.open(file_1))
    hash2 = imagehash.phash(Image.open(file_2))

    diff = hash1 - hash2
    print(diff)
    if diff >7:
        return False
    return True
class ImagePanel(wx.Panel):

    def __init__(self, parent):
        super().__init__(parent)
        self.max_size = 540
        self.photos = []
        self.current_photo = 0
        self.total_photos = 0
        self.layout()
        self.TimeFlick = 200 #ms
        self.slideshow_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_next, self.slideshow_timer)

        self.dataframe = []
        self.boundingbox = []
        self.isRightMouseClick = False
        self.btn_sizer
        self.isLeftMouseClick = False

        #slider
        self.sld
        self.slider_sizer
        #progressbar
        self.progress_label
        self.progressbar_sizer
        self.progress_bar
        ### for statistic
        self.total_corrected = 0
        self.list_photo_corrected = []

        ##imagehash
        self.previous_frame_corrected = False
        self.image_file_corrected = ""
        self.boundingbox_of_image_corrected = []





    def layout(self):
        """
        Layout the widgets on the panel
        """

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        #image frame
        img = wx.Image(self.max_size, self.max_size)
        self.image_ctrl = wx.StaticBitmap(self, wx.ID_ANY,
                                             wx.Bitmap(img))

        self.image_ctrl.Bind(wx.EVT_RIGHT_UP, self.onRIGHTClick)
        self.image_ctrl.Bind(wx.EVT_MOTION, self.onDraging)

        self.image_ctrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDownClick)
        self.image_ctrl.Bind(wx.EVT_LEFT_UP, self.onLeftUpClick)
        self.main_sizer.Add(self.image_ctrl, 0, wx.ALL | wx.CENTER, 5)
        #end image frame

        # progress bar
        self.progressbar_sizer = wx.BoxSizer(wx.HORIZONTAL)


        self.progress_bar = wx.Gauge(self, wx.ID_ANY, 10000,  size = wx.Size(200, 15),
               style =  wx.GA_HORIZONTAL | wx.GA_SMOOTH)
        self.progress_bar.SetValue(0)
        self.progress_label = wx.StaticText(self, label="Browsing progress 0%")
        self.progressbar_sizer.Add(self.progress_label, 0, wx.ALL | wx.CENTER, 5)
        self.progressbar_sizer.Add(self.progress_bar,  0, wx.ALL | wx.CENTER, 5)
        self.main_sizer.Add(self.progressbar_sizer, 0, wx.ALL|wx.CENTER, 5)
        # end progress bar

        #image label
        self.image_label = wx.StaticText(self, label="* Frame: 0\n* Total Frames Corrected: 0")

        self.main_sizer.Add(self.image_label, 0, wx.ALL|wx.CENTER, 5)
        #end image label


        # slider

        self.slider_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.sld = wx.Slider(self, value= 200, minValue= 50, maxValue= 500,
                        style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS, size=wx.Size(200, 50))
        self.sliderLabel = wx.StaticText(self, label="Adjust Frame Rate ({} FPS):".format(int(1000/self.sld.GetValue())))
        self.slider_sizer.Add(self.sliderLabel, 0, wx.ALL | wx.CENTER, 5)
        self.slider_sizer.Add(self.sld, 0, wx.ALL | wx.CENTER, 5)
        self.main_sizer.Add(self.slider_sizer, 0, flag=wx.ALL | wx.CENTER, border=10)
        self.sld.Bind(wx.EVT_SLIDER, self.onSliderScroll)

        # end slider

        #buttons
        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_data = [
                    ("Previous", self.btn_sizer, self.on_previous),
                    ("Slide Show", self.btn_sizer, self.on_slideshow),
                    ("Next", self.btn_sizer, self.on_next_new),
                    ("Delete", self.btn_sizer, self.on_delete)
        ]
        for data in btn_data:
            label, sizer, handler = data
            self.btn_builder(label, sizer, handler)

        self.main_sizer.Add(self.btn_sizer, 0, wx.CENTER)
        #end buttons

        self.SetSizer(self.main_sizer)

    def onSliderScroll(self, e):
        obj = e.GetEventObject()
        val = obj.GetValue()
        self.TimeFlick = val
        self.slideshow_timer.Start(self.TimeFlick)
        self.btn_sizer.Children[1].Window.SetLabel("Stop")
        self.sliderLabel.SetLabel("Adjust Frame Rate ({} FPS):".format(int(1000/self.sld.GetValue())))


    def btn_builder(self, label, sizer, handler):
        """
        Builds a button, binds it to an event handler and adds it to a sizer
        """
        btn = wx.Button(self, label=label)
        btn.Disable()
        btn.Bind(wx.EVT_BUTTON, handler)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)


    def on_delete(self, event):

        #print(self.current_photo)
        #print(self.photos[self.current_photo])
        photo_removed = self.photos[self.current_photo]
        #print(self.dataframe)
        #drop row in dataframe with the row number (index) accordingly
        row_index = self.dataframe.index[self.current_photo]
        self.dataframe= self.dataframe.drop(row_index)
        #print("after drop")
        #print(self.dataframe)
        self.photos.remove(photo_removed)
        self.total_photos = len(self.photos)
        if len(self.photos) == 0:
            self.btn_sizer.Children[0].Window.Disable()
            self.btn_sizer.Children[1].Window.Disable()
            self.btn_sizer.Children[2].Window.Disable()
            self.btn_sizer.Children[3].Window.Disable()
            img = wx.Image(self.max_size, self.max_size)
            self.image_ctrl.SetBitmap(wx.Bitmap(img))
            self.image_label.SetLabel("* Frame: 0"
                                      + "\n* Total Frames Corrected: 0")
            return

        if not self.photos:
            return

        if self.current_photo == self.total_photos - 1:
            self.current_photo = 0
        else:
            self.current_photo += 1
        if self.current_photo > self.total_photos:
            self.current_photo = 0

        self.update_photo(self.photos[self.current_photo], self.dataframe.iloc[self.current_photo][4:8].tolist())

        #update text label
        if photo_removed in self.list_photo_corrected:
            self.list_photo_corrected.remove(photo_removed)
        self.total_corrected = len(self.list_photo_corrected)
        # updalte text shown
        photo_file = os.path.basename(self.photos[self.current_photo])

        self.image_label.SetLabel("* Frame: " + str(self.current_photo + 1) + "/" + str(self.dataframe.shape[0])
                                  + "(" + photo_file + ")"
                                  + "\n* Total Frames Corrected: {}".format(self.total_corrected))

    def on_next(self, event):
        self.isRightMouseClick = False
        """
        Loads the next picture in the directory
        """
        if not self.photos:
            return

        if self.current_photo == self.total_photos - 1:
            self.current_photo = 0
        else:
            self.current_photo += 1
        if self.current_photo > self.total_photos:
            self.current_photo = 0
        #print(self.dataframe)
        self.update_photo(self.photos[self.current_photo], self.dataframe.iloc[self.current_photo][4:8].tolist())


    def on_next_new(self, event):
        self.isRightMouseClick = False
        """
        Loads the next picture in the directory
        """
        if not self.photos:
            return

        if self.current_photo == self.total_photos - 1:
            self.current_photo = 0
        else:
            self.current_photo += 1
        if self.current_photo > self.total_photos:
            self.current_photo = 0
        #print(self.dataframe)
        if self.previous_frame_corrected and two_image_is_identical(self.image_file_corrected, self.photos[self.current_photo]):
            self.update_photo(self.photos[self.current_photo], self.boundingbox_of_image_corrected)
            self.previous_frame_corrected = True
            photo_file = os.path.basename(self.photos[self.current_photo])
            # print("photo fille" + photo_file)
            index_of_photo_file_in_dataframe = self.dataframe[self.dataframe['filename'] == photo_file].index.item()

            # print(index_of_photo_file_in_dataframe)
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'xmin'] = self.boundingbox[0]
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'ymin'] = self.boundingbox[1]
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'xmax'] = self.boundingbox[2]
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'ymax'] = self.boundingbox[3]
            print("last image is corrected manually, and this image identical with the previous, so use the previous boundingbox")
        else:
            print("no identical")
            self.previous_frame_corrected = False
            self.update_photo(self.photos[self.current_photo], self.dataframe.iloc[self.current_photo][4:8].tolist())


    def on_previous(self, event):
        """
        Displays the previous picture in the directory
        """
        if not self.photos:
            return

        if self.current_photo == 0:
            self.current_photo = self.total_photos - 1
        else:
            self.current_photo -= 1
        self.update_photo(self.photos[self.current_photo], self.dataframe.iloc[self.current_photo][4:8].tolist())

    def on_speedup(self, event):
        self.TimeFlick = self.TimeFlick/2
        #self.slideshow_timer.Start(self.TimeFlick)
    def on_slideshow(self, event):
        """
        Starts and stops the slideshow
        """
        btn = event.GetEventObject()
        label = btn.GetLabel()
        if label == "Slide Show":
            self.slideshow_timer.Start(self.TimeFlick)
            btn.SetLabel("Stop")
        else:
            self.slideshow_timer.Stop()
            btn.SetLabel("Slide Show")

    def update_photo(self, image, boundingbox, t = 0):
        """
        Update the currently shown photo
        """

        img = wx.Image(image, wx.BITMAP_TYPE_ANY)



        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()

        # if W > H:
        #     NewW = self.max_size
        #     NewH = self.max_size * H / W
        # else:
        #     NewH = self.max_size
        #     NewW = self.max_size * W / H
        NewW = int(self.max_size)
        NewH = int(self.max_size * H/W)

        cv2_img = cv2.imread(image)
        xmin = -1
        ymin = -1
        xmax = -1
        ymax = -1
        if t == 0:
           xmin = boundingbox[0]
           ymin = boundingbox[1]
           xmax = boundingbox[2]
           ymax = boundingbox[3]
        elif t == 1 or t == 2:
            xmin = int(boundingbox[0]*W/NewW)
            ymin = int(boundingbox[1]*H/NewH)
            xmax = int(boundingbox[2]*W/NewW)
            ymax = int(boundingbox[3]*H/NewH)
            if t == 1:
                self.boundingbox = [xmin, ymin, xmax, ymax]


        cv2_img = cv2.rectangle(cv2_img, (xmin, ymin), (xmax, ymax),
                                    (255, 255, 0), 2)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img = cv2.resize(cv2_img,(NewW, NewH))
        #cv2_img = cv2.rectangle(cv2_img, ((int(boundingbox[0]*NewW/W), int(boundingbox[1]*NewH/H))), (int(boundingbox[2]*NewW/W), int(boundingbox[3]*NewH/H)),
        #                        (255, 255, 0), 1)

        h, w = cv2_img.shape[:2]


        # make a wx style bitmap using the buffer converter
        wxImage = wx.Bitmap.FromBuffer(w, h, cv2_img)

        self.image_ctrl.SetBitmap(wx.Bitmap(wxImage))
        photo_file = os.path.basename(self.photos[self.current_photo])
        self.image_label.SetLabel("* Frame: "  + str(self.current_photo + 1) + "/" + str(self.dataframe.shape[0])
                                + "(" + photo_file + ")"
                                  + "\n* Total Frames Corrected: {}".format(self.total_corrected))
        self.progress_label.SetLabel("Browsing progress {}%".format(int(self.current_photo  / len(self.photos) *100)))

        self.Refresh()
        self.thiscurrentphotonumber = self.current_photo

        self.progress_bar.SetValue(self.progress_bar.GetRange() * self.current_photo / len(self.photos))

        if self.current_photo == self.dataframe.shape[0] -1:
            self.progress_bar.SetValue(self.progress_bar.GetRange())
            self.progress_label.SetLabel("Browsing progress 100%")
            self.btn_sizer.Children[1].Window.SetLabel("Slide Show")

            self.slideshow_timer.Stop()




    def reset(self):
        img = wx.Image(self.max_size,
                       self.max_size)
        bmp = wx.Bitmap(img)
        self.image_ctrl.SetBitmap(bmp)
        self.current_photo = 0
        self.photos = []



    def onRIGHTClick(self, event):

        label = self.btn_sizer.Children[1].Window.GetLabel()
        if len(self.photos)!=0:
            self.image_ctrl.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        if label == "Stop":
            self.slideshow_timer.Stop()
            self.btn_sizer.Children[1].Window.SetLabel("Slide Show")

        if len(self.dataframe)!= 0:
            self.isRightMouseClick = True
            print("Right click on the photo panel")
            # TODO will go into the boundingbox editing mode
            # self.BackgroundColour = wx.GREEN
            self.boundingbox = []
            if self.photos:
                self.update_photo(self.photos[self.current_photo], [0, 0, 0, 0])

    def onDraging(self, event):

        if self.isLeftMouseClick:
            print("left mouse down and drag")


            pt = event.GetPosition()

            xmax = pt[0]
            ymax = pt[1]


            xmin = self.boundingbox[0]
            ymin = self.boundingbox[1]
            if xmax < xmin:
                temp = xmax
                xmax = xmin
                xmin = temp
            if ymax < ymin:
                temp = ymax
                ymax = ymin
                ymin = temp

            self.update_photo(self.photos[self.current_photo], (xmin, ymin, xmax, ymax), 2)


    def onLeftDownClick(self, event):

        if self.isRightMouseClick:
            self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
            self.isLeftMouseClick = True
            pt = event.GetPosition()
            xmin = pt[0]
            ymin = pt[1]
            self.boundingbox.append(xmin)
            self.boundingbox.append(ymin)
            #print("Position of mouse left-down-click")
            #print(self.boundingbox)
            #print("----")
            #print(pt)


    def onLeftUpClick(self, event):
        self.isLeftMouseClick = False
        if self.isRightMouseClick:
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
            pt = event.GetPosition()
            #print(pt)
            xmax = pt[0]
            ymax = pt[1]
            if xmax < self.boundingbox[0]:
                temp = xmax
                xmax = self.boundingbox[0]
                self.boundingbox[0] = temp
            if ymax < self.boundingbox[1]:
                temp = ymax
                ymax = self.boundingbox[1]
                self.boundingbox[1] = temp
            self.boundingbox.append(xmax)
            self.boundingbox.append(ymax)

            self.update_photo(self.photos[self.current_photo], self.boundingbox, 1)
            #print(self.boundingbox)
            #print("before change bounding box")
            #print(self.dataframe)
            #print("current photo number " + str(self.current_photo))
            photo_file = os.path.basename(self.photos[self.current_photo])
            #print("photo fille" + photo_file)
            index_of_photo_file_in_dataframe = self.dataframe[self.dataframe['filename'] == photo_file].index.item()

            #print(index_of_photo_file_in_dataframe)
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'xmin'] = self.boundingbox[0]
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'ymin'] = self.boundingbox[1]
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'xmax'] = self.boundingbox[2]
            self.dataframe.loc[index_of_photo_file_in_dataframe, 'ymax'] = self.boundingbox[3]
            #print("after change bounding box")
            #print(self.dataframe)
            self.isRightMouseClick = False
            self.previous_frame_corrected = True
            self.image_file_corrected = self.photos[self.current_photo]
            self.boundingbox_of_image_corrected = self.boundingbox
            if len(self.list_photo_corrected) == 0 or self.photos[self.current_photo] not in self.list_photo_corrected:
                self.list_photo_corrected.append(self.photos[self.current_photo])
                self.total_corrected = len(self.list_photo_corrected)
                #updalte text shown
                self.image_label.SetLabel("* Frame: " + str(self.current_photo + 1) + "/" + str(self.dataframe.shape[0])
                                          + "("+photo_file + ")"
                                          + "\n* Total Frames Corrected: {}".format(self.total_corrected))
                #print(self.list_photo_corrected)




class MainFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, title='Bounding Box Inspection Window',
                         size=(900, 900))
        self.panel = ImagePanel(self)
        self.create_toolbar()
        self.Show()
        self.Bind(wx.EVT_CLOSE, self.OnExit)
        self.csv_file = ""
        self.frame_folder = "../data/frame/"



    def OnExit(self, event):
        #print(self.panel.dataframe)
        if event.CanVeto():
            if len(self.panel.dataframe)!= 0:
                if wx.MessageBox("Saving updated data before closing?",
                                 "Please confirm",
                                 wx.ICON_QUESTION | wx.YES_NO) == wx.YES:
                    print("All data is saved!!!", self.csv_file)
                    #TODO saved all data from dataframe to file
                    #print(self.dataframe)
                    self.panel.dataframe.to_csv(self.csv_file, index=False)
                    #event.Veto()
                    #return

                else:
                    print("Data is not saved!!")
                    #event.Veto()
                    #return


        self.Destroy()




    def create_toolbar(self):
        """
        Create a toolbar
        """
        self.toolbar = self.CreateToolBar()
        self.toolbar.SetToolBitmapSize((32,32))

        open_ico = wx.ArtProvider.GetBitmap(
            wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (32,32))
        openTool = self.toolbar.AddTool(
            wx.ID_ANY, "Open", open_ico, "Select csv file result")
        self.Bind(wx.EVT_MENU, self.on_open_csv, openTool)

        self.toolbar.Realize()

    def on_open_directory(self, event):
        """
        Open a directory dialog
        """

        with wx.DirDialog(self, "Choose a directory",
                          style=wx.DD_DEFAULT_STYLE) as dlg:

            if dlg.ShowModal() == wx.ID_OK:
                self.folder_path = dlg.GetPath()

                photos = glob.glob(os.path.join(self.folder_path, '*.jpg'))
                self.panel.photos = photos
                if photos:
                    self.panel.update_photo(photos[0])
                    self.panel.total_photos = len(photos)

                else:
                    self.panel.reset()


    def on_open_csv(self, event):
        """
        Open a csv file
        """
        if len(self.panel.dataframe)!= 0:
            if wx.MessageBox("Saving updated data before selecting another csv result?",
                             "Please confirm",
                             wx.ICON_QUESTION | wx.YES_NO) == wx.YES:
                print("All data is saved!!!", self.csv_file)
                # TODO saved all data from dataframe to file
                # print(self.dataframe)
                self.panel.dataframe.to_csv(self.csv_file, index=False)
                # event.Veto()
                # return

            else:
                print("Data is not saved!!")




        with wx.FileDialog(self, "Choose a csv file to browse the result",
                          style=wx.DD_DEFAULT_STYLE) as dlg:
        #with wx.DirDialog(self, "Choose a csv file to browse the result",
        #                  style=wx.DD_DEFAULT_STYLE) as dlg:

            if dlg.ShowModal() == wx.ID_OK:
                self.csv_file = dlg.GetPath()
                #csv_file = self.folder_path
                print("File chosen "+ self.csv_file)
                #TODO: ignore file chosen if it is not a csv file


                self.panel.dataframe = pd.read_csv(self.csv_file)

                #Take link to the frame folder
                obj = os.path.basename(dirname(Path(self.csv_file)))
                step = Path(self.csv_file).stem
                #print(obj)
                photo_directory = os.path.join(self.frame_folder,obj, step)
                if step == "test":
                    photo_directory = "../data/frame/0wheel/"


                photo_list = []
                photo_filename = self.panel.dataframe['filename'].tolist()
                print(photo_filename)
                for i in photo_filename:
                    # print(os.path.join( photo_directory, i))
                    photo_list.append(os.path.join(photo_directory, i))

                self.panel.photos = photo_list

                if photo_list:
                    self.panel.current_photo = 0
                    self.panel.update_photo(photo_list[0], self.panel.dataframe.iloc[0].tolist()[4:8])
                    self.panel.total_photos = len(photo_list)
                    self.panel.btn_sizer.Children[0].Window.Enable()
                    self.panel.btn_sizer.Children[1].Window.Enable()
                    self.panel.btn_sizer.Children[2].Window.Enable()
                    self.panel.btn_sizer.Children[3].Window.Enable()

                    self.panel.list_photo_corrected = []
                    self.panel.total_corrected = 0
                    self.panel.current_photo = 0
                    self.panel.image_label.SetLabel(
                        "* Frame: " + str(self.panel.current_photo + 1) + "/" + str(self.panel.dataframe.shape[0])
                        + "\n* Total Frames Corrected: {}".format(self.panel.total_corrected))

                else:
                    self.panel.reset()
        #print(self.dataframe)






if __name__ == '__main__':
    app = wx.App(redirect=False)
    frame = MainFrame()
    app.MainLoop()