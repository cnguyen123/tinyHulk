# tinyHulk --auto_labelling tool

**1. Introduction**

The program automatically annotates objects in the input _green-background frames_; generate new frames with different backgrounds from the original frames; generate tfrecord data file for the annotated frames; and split the tfrecord data file (train, test, validation data files) for the purspose of training the object detection model.  

**2. Package Installation**
(Prefer to install in a virtual environment. For example, in pycharm Press Ctrl+Alt+S to open the IDE settings and select Project <project name> | Python Interpreter.

Create a virtual environment using the project requirements. Activate that new environment (source venv/bin/activate), then install the packages below:)
- opencv: pip install opencv-contrib-python
- pandas: pip install pandas
- tensorflow: pip install tensorflow
- tensorflow-object-detection: pip install tensorflow-object-detection-api
- install wxpython: (For unbuntu, check what version of your ubuntu to download and install a proper package. Below is command to install wxPython for ubuntu 18.04: pip install -U https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl .)
   If error libSDL2-2.0.so.0: cannot open shared object file: No such file or directory. Install sudo apt-get install git curl libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0

**3. How to Use**

- Step 1: Run **_video_to_frame.py_** to parse frames from an input video (.mp4 format).
- Step 2: Annotate frames with **_draw_box.py_**
- Step 3: Create tfrecord for the annotated frames with **_tfrecord_create.py_** (1.tfrecord)
- Step 4: Generate new data by changing background of the original frames with _**greenbackground_changing.py**_
   
   &nbsp;It first needs to decide a proper value of _lower_green_ and _upper_green_ of the HSV (Hue, Saturation, Value) color model. User tunes these &nbsp;values from the pop-up pannel and sees the result showing in the photo. 
   
   
- Step 5: Create tfrecord for the new background frames with **_tfrecord_create.py_** (2.tfrecord)
- Step 6: Merge the two tfrecords (1.tfrecord, 2.tfrecord) created above with _**merge_tfrecords.py**_
- Step 7: Spit the merged_tfrecord to train.tfrecord, test.tfrecord, val.tfrecord with _**tfrecord_split.py**_
