import json
import tensorflow as tf
import os

def test_cifar(meta_filename, model_filename, epochs, batch_size):
    import cPickle
    import numpy as np
    with open(meta_filename) as fd:
        model_meta = json.loads(fd.read())

    graph_def = tf.GraphDef()

    with open(model_filename) as fd:
        graph_def.ParseFromString(fd.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=graph) as sess:
        sess.run('init')
        for epoch in range(epochs):
            for batch in range(1, 6):
                with open('/datasets/cifar-10-batches-py/data_batch_%d' % batch, 'rb') as fd:
                    d = cPickle.load(fd)

                    x_batch = []
                    y_batch = []
                    for image, label in zip(d['data'], d['labels']):
                        image = np.array(image)
                        x_batch.append(image)
                        y_batch.append([label])
                        if len(y_batch) == batch_size:
                            sess.run(model_meta['train_op'], feed_dict={
                                          model_meta['inputs']: x_batch,
                                          model_meta['labels']: y_batch})
                            print(sess.run(model_meta['total_loss']))
                            x_batch = []
                            y_batch = []


def test_mnist(meta_filename, model_filename, epochs=3, batch_size=64):

    graph_def = tf.GraphDef()

    with open(model_filename) as fd:
        graph_def.ParseFromString(fd.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=graph) as sess:
        meta_json =sess.run("meta")
    print(meta_json)
    with open(meta_filename) as fd:
        model_meta = json.loads(meta_json)

    name, _ = os.path.splitext(meta_filename)
    name = os.path.split(name)[-1]
    test_mnist_with_graph(name, graph, model_meta, epochs=epochs, batch_size=batch_size)


def test_mnist_with_graph(name, graph, model_meta, epochs, batch_size):
    import mnist

    x_test_batch = []
    y_test_batch = []

    for image, label in zip(mnist.dataset.test_images,
                            mnist.dataset.test_labels):
        label_onehot = [0]*10
        label_onehot[label] = 1.0
        x_test_batch.append(image)
        y_test_batch.append(label_onehot)

    if batch_size == -1:
        batch_size = len(x_test_batch)

    import datetime
    dt = datetime.datetime.now()
    now = "{:%Y%m%d%H%M%s}".format(dt)
    summary_dir = "/tmp/test_mnist/"+name+"/" + now
    print("summary dir:", summary_dir)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        sess.run('init')
        sw = tf.train.SummaryWriter(summary_dir, graph=sess.graph)
        global_step = 0
        for epoch in xrange(epochs):
            x_batch = []
            y_batch =[]
            count = 0
            for image, label in zip(
               mnist.dataset.train_images, mnist.dataset.train_labels):
               count += 1
               label_onehot = [0]*10
               label_onehot[label] = 1.0

               x_batch.append(image)
               y_batch.append(label_onehot)

               assert model_meta['kind'] == 'ImageCategoryPrediction'
               if len(y_batch) == batch_size:
                   feed_dict = {
                      model_meta['inputs']['batch_image_input']: x_batch,
                      model_meta['inputs']['categorical_labels']: y_batch,
                      model_meta['parameters']['learning_rate']: 0.045,
                      model_meta['parameters']['global_step']: global_step
                   }

                   _, accuracy, loss, global_step, lr = sess.run(
                       [
                           model_meta['train_op'],
                           model_meta['metrics']['accuracy'],
                           model_meta['metrics']['total_loss'],
                           model_meta['parameters']['global_step'],
                           model_meta['parameters']['learning_rate']
                       ],
                       feed_dict=feed_dict
                   )

                   print("train accuracy:%f, total_loss:%f, step: %d, lr: %f" % (
                           accuracy,
                           loss,
                           global_step, lr))

                   global_step += 1
            x_batch = []
            y_batch = []

            # test Error at the end of the epoch
            total_precision = 0
            total_loss = 0
            batches = 0
            for i in range(0, len(x_test_batch), batch_size):
                feed_dict = {
                model_meta['inputs']['batch_image_input']: x_test_batch[i:i+batch_size],
                model_meta['inputs']['categorical_labels']: y_test_batch[i:i+batch_size],
                }

                precision, loss = sess.run(
                [
                    model_meta['metrics']['accuracy'],
                    model_meta['metrics']['total_loss']
    #                           model_meta['summary_op'],
    #                           model_meta['parameters']['global_step']
                ],
                feed_dict=feed_dict)

                #sw.add_summary(summaries)
                #sw.flush()
                total_loss += loss
                total_precision += precision
                batches += 1

            print("test accuracy:%f, total_loss:%f" % (
                total_precision/float(batches),
                total_loss/float(batches),
                ))

            print("epoch: %d" % epoch)
    sw.close()
