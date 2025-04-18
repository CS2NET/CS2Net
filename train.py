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



def distribute_train(FLAGS_):
    global FLAGS
    FLAGS = FLAGS_

    tf.logging.info("job name = %s" % FLAGS.job_name)
    tf.logging.info("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_count = len(worker_spec)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    train(worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
          target=server.target)


def train(worker_count, task_index, cluster, is_chief, target):
    worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
    tf.logging.info("worker_deivce = %s" % worker_device)

    model_dir_restore = FLAGS.model_dir_restore
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    checkpointDir = FLAGS.checkpointDir
    buckets = FLAGS.buckets
    model_dir = os.path.join(checkpointDir, model_dir)
    model_dir_restore = os.path.join(checkpointDir, model_dir_restore)
    tf.logging.info(
        "buckets:{} checkpointDir:{} checkpointDir_restore:{}".format(buckets, model_dir, model_dir_restore))
    # -----------------------------------------------------------------------------------------------
    tf.logging.info("loading input...")
    train_file = FLAGS.train_tables.split(',')

    with tf.device(worker_device):
        train_dataset = input_fn(train_file, batch_size, 'train', is_sequence_train=FLAGS.is_sequence_train,
                                 slice_count=worker_count, slice_id=task_index)
        train_iterator = train_dataset.make_one_shot_iterator()
    tf.logging.info("finished loading input...")

    available_worker_device = "/job:worker/task:%d" % (task_index)
    with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        loss, train_op, train_metrics = train_model_fn(train_iterator, global_step)

    tf.logging.info("start training")
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    for var in bn_moving_vars:
        tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_train_step)]
    hooks_for_chief = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=model_dir,
            save_secs=FLAGS.save_time,
            saver=tf.train.Saver(name='chief_saver'))
    ]

    if FLAGS.is_restore:
        ckpt_state = tf.train.get_checkpoint_state(model_dir_restore)
        if not (ckpt_state and tf.train.checkpoint_exists(ckpt_state.model_checkpoint_path)):
            tf.logging.info("restore path error!!!")
            raise ValueError
    else:
        model_dir_restore = None

    step = 0

    with tf.train.MonitoredTrainingSession(checkpoint_dir=model_dir_restore,
                                           master=target,
                                           is_chief=is_chief,
                                           config=sess_config,
                                           save_checkpoint_secs=None,
                                           hooks=hooks,
                                           chief_only_hooks=hooks_for_chief) as sess:

        chief_is_end = False
        sess_is_end = False
        while (not sess_is_end) and (not sess.should_stop()):
            if not chief_is_end:
                try:
                    step += 1
                    global_step_val = train_step(loss, train_op, train_metrics, step, global_step, sess, "step")
                except tf.errors.OutOfRangeError as e:
                    if is_chief:
                        tf.logging.info("chief node end...")
                        chief_is_end = True
                        tf.logging.info("waiting all worker nodes to be end")
                        last_step = global_step_val
                    else:
                        tf.logging.info("worker node end...")
                        break
            else:
                while 1:
                    time.sleep(60)
                    tf.logging.info("waiting all worker nodes to be end")
                    global_step_val = sess.run(global_step)
                    if global_step_val > last_step:
                        last_step = global_step_val
                    else:
                        tf.logging.info("all worker nodes end. chief node is finished")
                        sess_is_end = True
                        break
    tf.logging.info("%d steps finished." % step)
