import tensorflow as tf, sys

image_path = sys.argv[1]

# setup
label_file = 'retrained_labels.txt'
graph_file = 'retrained_graph.pb'

# Read in image
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

label_lines = [line.rstrip() for line in tf.gfile.GFile(label_file)]

# open retrained graph file and parse the model
with tf.gfile.FastGFile(graph_file, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

# start a tensorflow session
with tf.Session() as sess:

	# Input image data and get first prediction. Use softmax to get prediction
	# maps the final layer into probabilities for each classifier
	softmax = sess.graph.get_tensor_by_name('final_result:0')

	# executing softmax tensor function on input image
	predictions = sess.run(softmax, {'DecodeJpeg/contents:0': image_data})

	# grab label in order of confidence. Reverse sorted array to grab max
	in_order = predictions[0].argsort()[-len(predictions[0]):][::-1]
	match_label = label_lines[in_order[0]]
	match_score = predictions[0][in_order[0]] * 100
	print ('-----------------------------------------------------')
	print('Welcome stranger...\n')
	print('Seems like you most resemble {0} from the fellowship.\n'.format(match_label))
	print("Let me see... I predict you're {0} with a {1}% chance!".format(match_label, round(match_score, 3)))
	print ('-------------------Other Members---------------------')
	print ('details:')
	for node_id in in_order[1:]:
		classifier = label_lines[node_id]
		score = predictions[0][node_id]
		print('%s (score = %.5f)' % (classifier, score))