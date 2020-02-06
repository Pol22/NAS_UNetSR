
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import save_model

import argparse
import numpy as np

from queue import Queue
import threading
import subprocess
from random import choice
from time import time

from blocks import generate_all_possible_blocks, random_select_blocks
from blocks import str_to_block
from UNet import UNet, how_many_blocks
from evaluation import Evaluator


TRAIN_SCRIPT = 'train.py'
TRAINING_LOG_FILE = 'train.log'
MODEL_NAME_SUFFIX = '.h5'
BLOCKS_DIVISOR = '+'


class Trainer(threading.Thread):
    '''
        Deamon thread for training
    '''
    def __init__(self, queue, gpu_index):
        super(Trainer, self).__init__()
        self.queue = queue
        self.gpu_index = gpu_index

    def run(self):
        while True:
            model_path = self.queue.get()
            self.train(model_path)
            self.queue.task_done()

    def train(self, model_path):
        print('Start training model:\n' + model_path)
        log_file = model_path + '.csv'
        args = ['python', TRAIN_SCRIPT, '--model', model_path, '--gpu',
                str(self.gpu_index), '--output', model_path, '--log_file',
                log_file]

        start = time()
        process = subprocess.run(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        end = time()
        print(f'Elapsed time {(end - start)/60:04.2f}min ' + \
              f'for training model:\n{model_path}')


def model_path_to_blocks_list(model_path):
    model_name = os.path.split(model_path)[1] # (head, tail) -> tail
    if model_name.endswith(MODEL_NAME_SUFFIX):
        model_blocks = model_name.split(MODEL_NAME_SUFFIX)[0]
        model_blocks = model_blocks.split(BLOCKS_DIVISOR)
        result = list()
        for str_block in model_blocks:
            result.append(str_to_block(str_block))
        
        return result
    else:
        raise Exception('Wrong model path: {}'.format(model_path))


def blocks_list_to_model_path(folder, blocks):
    model_name = BLOCKS_DIVISOR.join(map(str, blocks)) + MODEL_NAME_SUFFIX
    model_path = os.path.join(folder, model_name)
    return model_path


def train_children(childred, img_size, queue, train_folder):
    '''
        Return list of trained model paths
    '''
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    trained_models = list()
    for child in childred:
        # create model from child's blocks
        model = UNet((img_size, img_size, 3), child)
        model_path = blocks_list_to_model_path(train_folder, child)
        trained_models.append(model_path)
        save_model(model, model_path, include_optimizer=False)
        # put model on queue for training
        queue.put(model_path)
    # wait training of all models
    queue.join()

    return trained_models

# TODO create shared blocks between models
class MutationController(object):
    def __init__(self, blocks, blocks_length, mutation_prob, alpha, eps):
        self.blocks = blocks
        self.mutation_prob = mutation_prob
        self.alpha = alpha
        self.eps = eps
        self.block_credit = np.zeros(shape=(len(blocks), blocks_length))

    def _uniform_mutation(self, child):
        mutation_rand = np.random.uniform(size=len(child)) # rand [0.0, 1.0)
        for i, _ in enumerate(child):
            if mutation_rand[i] < self.mutation_prob:
                child[i] = choice(self.blocks) # select uniform random block

        return child

    def _roulette_wheel_selection(self, block_index):
        index_credits = self.block_credit[:, block_index] # credits by depth
        # normalization
        norm_credits = index_credits - \
            (np.min(self.block_credit, axis=1) - self.eps)
        # probs
        probs = norm_credits / np.sum(norm_credits)
        cum_sum = np.cumsum(probs)
        rand_num = np.random.uniform()
        rand_index = np.sum(cum_sum < rand_num)

        return self.blocks[rand_index]

    def guided_mutation(self, elite):
        model_len = len(elite[0][1]) # [(PSNR, blocks), ...] -> len(blocks)
        child = list()
        for block_ind in range(model_len):
            child.append(self._roulette_wheel_selection(block_ind))
        # use uniform mutation
        return self._uniform_mutation(child)

    def crossover(self, elite):
        '''
            Crossover between elites
            (Random select blocks from elites
            with uniform mutation after)
        '''
        model_len = len(elite[0][1]) # [(PSNR, blocks), ...] -> len(blocks)
        elite_len = len(elite)
        child = list()
        # rand choice parent for each block
        rand_ind = np.random.randint(elite_len, size=model_len)
        for block_ind in range(model_len):
            selected_elite = elite[rand_ind[block_ind]]
            elite_blocks = selected_elite[1] # (PSNR, blocks)
            child.append(elite_blocks[block_ind])
        # use uniform mutation
        return self._uniform_mutation(child)

    def update(self, evaluations):
        '''
            Update block credits from evaluations
        '''

        for psnr, model_path in evaluations:
            blocks = model_path_to_blocks_list(model_path)
            for i, block in enumerate(blocks):
                index = self.blocks.index(block)
                # update block credit
                if self.block_credit[index, i] != 0:
                    self.block_credit[index, i] *= self.alpha
                    self.block_credit[index, i] += (1 - self.alpha) * psnr
                else:
                    self.block_credit[index, i] = psnr

        # with open(TRAINING_LOG_FILE, 'a') as log:
        #     log.write('\nUpdated block credit!\n')
        #     log.write(str(self.block_credit))
        #     log.write('\n')
        
        print('\nBlock credits updated!\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, help='img height/width',
                        default=64)
    parser.add_argument('--unet_depth', type=int,
                        help='how many times used downsampling',
                        default=3)
    parser.add_argument('--batch', type=int, help='batch size',
                        default=64)
    parser.add_argument('--generations', type=int,
                        help='number of generations',
                        default=40)
    parser.add_argument('--population', type=int,
                        help='population size',
                        default=16)
    parser.add_argument('--children', type=int,
                        help='number of children',
                        default=8)
    parser.add_argument('--mutation_prob', type=float,
                        help='mutation probability',
                        default=0.2)
    parser.add_argument('--elitism', type=int, help='elitism number',
                        default=8)
    parser.add_argument('--alpha', type=float,
                        help='block update coefficient',
                        default=0.9)
    parser.add_argument('--eps', type=float,
                        help='constant to normalize block credit',
                        default=0.001)
    parser.add_argument('--gpus', type=int, help='number of gpus',
                        default=8)
    parser.add_argument('--train_tmp', type=str,
                        help='temporary folder for training',
                        default='./train_tmp')
    parser.add_argument('--eval_data', type=str,
                        help='evaluation dataset',
                        default='./DIV2K/benchmark/B100')
    args = parser.parse_args()

    # evaluator creation
    evaluator = Evaluator(args.eval_data, args.batch, args.img_size)
    # initialization
    possible_blocks = generate_all_possible_blocks()
    # init population
    population = list()
    unet_block_num = how_many_blocks(args.unet_depth)
    for _ in range(args.population):
        population.append(
            random_select_blocks(possible_blocks, unet_block_num))
    # init mutation controller
    mutation_controller = MutationController(
        possible_blocks, unet_block_num,
        args.mutation_prob, args.alpha, args.eps)

    # init fitness
    fitnesses = list() # contain tuple (PSNR, list_of_blocks)
    for individual in population:
        fitnesses.append((0.0, individual))

    # queue for training models
    queue = Queue(args.gpus)
    # create training thread for each gpu
    for gpu_index in range(args.gpus):
        t = Trainer(queue, gpu_index)
        t.setDaemon(True)
        t.start()

    # generation steps
    for generation_num in range(args.generations):
        # get elite
        elite = fitnesses[:args.elitism]
        children = list()
        for _ in range(args.children // 2):
        # for _ in range(args.children):
            gm_child = mutation_controller.guided_mutation(elite)
            if not os.path.exists(blocks_list_to_model_path(
                                      args.train_tmp, gm_child)):
                children.append(gm_child)

            crossover_child = mutation_controller.crossover(elite)
            if not os.path.exists(blocks_list_to_model_path(
                                      args.train_tmp, crossover_child)):
                children.append(crossover_child)

        # train children
        models = train_children(children, args.img_size, queue,
                                args.train_tmp)
        # evaluate models
        pairs = evaluator.evaluate(models)
        # update mutation
        mutation_controller.update(pairs)
        # update fitness and log to file

        for psnr, model_path in pairs:
            # create list of blocks from model_path
            model = model_path_to_blocks_list(model_path)
            fitnesses.append((psnr, model))
            print('Eval [{:06.4f}] {}'.format(psnr, model_path))

        # sort fitnesses by PSNR
        fitnesses = sorted(fitnesses, key=lambda x: x[0], reverse=True)
        with open(TRAINING_LOG_FILE, 'a') as log:
            log.write('\nTop of generation {}\n'.format(generation_num))
            for psnr, model in fitnesses[:args.elitism]:
                model_path = blocks_list_to_model_path(args.train_tmp, model)
                log.write('[{:06.4f}] {}\n'.format(psnr, model_path))

        # remove weak models from tmp folder
        if len(fitnesses) > args.population:
            for _, model in fitnesses[args.population:]:
                model_path = blocks_list_to_model_path(args.train_tmp, model)
                if os.path.exists(model_path):
                    os.remove(model_path)
                # remove training log
                csv_path = model_path + '.csv'
                if os.path.exists(csv_path):
                    os.remove(csv_path)

            fitnesses = fitnesses[:args.population]


if __name__ == '__main__':
    main()
