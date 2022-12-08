import tensorflow as tf



#Read tf record from file
def decode_fm(record_bytes):
    return tf.io.parse_single_example(
        #Data
        record_bytes,

        #Schema
        {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
             #'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
    )

#data = "../data/tf_data/dummy2.tfrecord"
#data= "../data/merged_tf/something/merged.tfrecord"
data = "../data/val.tfrecord"
c = 0
for batch in tf.data.TFRecordDataset(data).map(decode_fm):
    c = c + 1
    a = '{image/filename}'.format(**batch)

    #if a != "b'frame04280.jpg'": continue
    print('image/height = {image/height:.4f}'.format(**batch))
    print('image/width = {image/width:.4f}'.format(**batch))
    print('image/filename = {image/filename}'.format(**batch))
    #print('image/source_id = {image/source_id}'.format(**batch))
    #print('image/encoded = {image/encoded}'.format(**batch))
    #print('image/format = {image/format}'.format(**batch))
    print('image/object/bbox/xmin = {image/object/bbox/xmin}'.format(**batch))
    print('image/object/bbox/xmax = {image/object/bbox/xmax}'.format(**batch))
    print('image/object/bbox/ymin = {image/object/bbox/ymin}'.format(**batch))
    print('image/object/bbox/ymax = {image/object/bbox/ymax}'.format(**batch))
    print('image/object/class/text = {image/object/class/text}'.format(**batch))
print(c)
