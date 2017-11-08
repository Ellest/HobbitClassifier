import os
import sys
import subprocess
from argparse import ArgumentParser

def train(steps):
	learning_rate = None#0.0001
	main_command = 'python'
	retrain_file = 'retrain.py'
	bottleneck_string = '--bottleneck_dir=bottlnecks'
	step_string = '--how_many_training_steps=' + str(steps)
	model_dir = '--model_dir=inception'
	graph_string = '--output_graph=retrained_graph.pb'
	label_string = '--output_labels=retrained_labels.txt'
	img_dir = '--image_dir=LOTR' 
	learning_rate_string = '--learning_rate=' + str(learning_rate)
	command_container = [
					main_command, 
					retrain_file,
					bottleneck_string, 
					step_string,
					model_dir, 
					graph_string, 
					label_string, 
					img_dir
				]
	if learning_rate:
		command_container.append(learning_rate_string)
	status = subprocess.call(command_container)
	if not status:
		print('training process successful')
	else:
		print('something went wrong with the training...')

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('usage: python trainer.py [number-of-steps] [learning-rate]')
	else:
		steps = sys.argv[1]
		if not steps.isdigit():
			print('arugment must be a number')
		else:
			print('running trainer with {0} steps...'.format(steps))
			train(steps)
