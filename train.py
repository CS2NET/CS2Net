# -*- coding: utf-8 -*-
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# train and dev
tf.flags.DEFINE_string('tables', '', '')
tf.flags.DEFINE_string('train_tables', '', '')
tf.flags.DEFINE_string('dev_tables', '', '')
tf.flags.DEFINE_string('outputs', '', '')

tf.flags.DEFINE_string('model_dir', 'model/', 'model dir')
tf.flags.DEFINE_string('model_dir_restore', 'model/', 'model dir')
tf.flags.DEFINE_bool('is_sequence_train', False, 'flag for train mode')
tf.flags.DEFINE_bool('is_restore', False, 'flag for restore')


tf.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.flags.DEFINE_integer('max_train_step', 1000000, 'max_train_step')
tf.flags.DEFINE_integer('dev_total', 100000, 'dev_total')
tf.flags.DEFINE_bool('is_training', True, '')
tf.flags.DEFINE_float('keep_prob', 1.0, "")

tf.flags.DEFINE_string('buckets', "", 'buckets')
tf.flags.DEFINE_string('checkpointDir', "", 'checkpointDir')

# transport
tf.flags.DEFINE_string('user_name', "your_name", "")
tf.flags.DEFINE_integer('model_version', 1, "")
tf.flags.DEFINE_string('model_name', "cs2net", "")

# distribute
tf.flags.DEFINE_integer("task_index", None, "Worker task index")
tf.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.flags.DEFINE_integer('aggregate', 100, 'aggregate batch number')
tf.flags.DEFINE_integer("save_time", 600, 'train epoch')

# mode and model selection
tf.flags.DEFINE_string('mode', 'train', "train/dev/transport")
tf.flags.DEFINE_string('model', 'cs2net', "model name")


FLAGS = tf.flags.FLAGS


# model selection and register
model_name = FLAGS.model
tf.logging.info("choose model: {}".format(model_name))
if model_name == 'cs2net':
    from CS2NET import *
  


def main(_):
    if FLAGS.mode == 'train':
        distribute_train(FLAGS)

