import copy
import csv
import gc
import logging

import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.initializer import Initializer
from src.model.student import Student
from src.model.trainer import Trainer
from src.utils import utils


class AutoCP(Initializer):
    """
    Entry point for AutoCP
    Code inspired from https://github.com/jacknewsom/autohas-pytorch
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.student_model = None

        self.trainer = None

        self.train_epochs = args.train_epochs
        self.argmax_epochs = args.argmax_epochs
        self.early_stop = args.early_stop
        self.early_stop_epoch = args.early_stop_epoch
        self.early_stop_acc = args.early_stop_acc
        self.eval_interval = args.eval_interval
        self.early_stop_no_impr = args.early_stop_no_impr
        self.warmup_epochs = args.warmup_epochs
        # number of rollouts in first iteration
        self.warmup_rollouts = args.warmup_rollouts
        # number of rollouts after
        self.num_rollouts_iter = args.rollouts
        # number of updates before saving controller policies
        self.save_policy_frequency = 1

        self.cp_used = self.args.dataset in {'cp19', 'cp29'}
        self.metric = 'auc' if self.cp_used else 'acc'
        logging.info("Optimizing controller towards {} metric.".format(self.metric))

    def train_controller(self):

        if self.args.cont_training:
            # get iteration from replay memory
            student_id = self.controller.replay_memory.__len__()
            assert student_id > 0
            # load action list
            from src.utils.io import read_actions
            action_path = "{}/action_list.csv".format(self.args.cont_dir)
            action_list = read_actions(action_path)
            argmax_id, argmax_index = max((i[0], index) for index, i in enumerate(action_list))
            argmax_id += 1
            # load rollouts
            from src.utils.io import read_rollouts
            rollouts_path = "{}/rollouts.csv".format(self.args.cont_dir)
            rollouts_save = read_rollouts(rollouts_path, self.gpu_id)
            logging.info("Loaded old actions and rollouts successfully.")

            iteration = argmax_id % 1000
            assert iteration < 2000, "You probably trained too long..."

            # check when argmax has to be trained
            if argmax_index + self.args.rollouts < len(action_list):
                train_argmax_bool = True
            else:
                train_argmax_bool = False
        else:
            iteration = 0
            student_id = 0
            argmax_id = 1000
            rollouts_save = []
            action_list = []
            train_argmax_bool = False

        logging.info("Training controller for {} dataset...".format(self.args.dataset))
        while not self.controller.has_converged():
            rollouts = []
            logging.info("Iteration {}".format(iteration))
            # warmup check
            if iteration == 0:
                rollout_num = self.warmup_rollouts
            else:
                rollout_num = self.num_rollouts_iter

            if train_argmax_bool:
                # skip training student
                rollout_num = 0
                train_argmax_bool = False
                logging.info("Due to many trained students now updating controller!")

            for t in range(rollout_num):
                logging.info("Rollout {}/{}".format(t, rollout_num))
                logging.info("Loading student ID: {}".format(student_id))

                # sample student
                model_params, hp_params, actions_arch, actions_hp = self.sample_student(student_id, argmax=False)

                # initialize trainer
                self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                                       self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                                       self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                                       self.new_batch_size)

                oom_bool = False
                epoch_no_improve = 0
                last_train_acc = 0.0
                student_time = 0
                if self.cp_used:
                    # more info for CP data
                    best_state_student = {'acc': 0, 'auc': 0, 'cm': 0, 'precision': 0, 'recall': 0, 'F1': 0, 'sens': 0,
                                          'spec': 0, 'ppv': 0, 'npv': 0, 'b_acc': 0}
                else:
                    best_state_student = {'acc': 0, 'acc5': 0, 'auc': 0, 'cm': 0}
                logging.info("Training student {} for {} epochs".format(student_id, self.train_epochs))
                try:
                    for epoch in range(self.train_epochs):
                        # student training loop
                        epoch_time, train_acc, train_loss = self.trainer.trainer_train(epoch, self.train_epochs,
                                                                                       student_id)

                        student_time += epoch_time
                        is_best = False
                        if (epoch + 1) % self.eval_interval == 0 or epoch == self.train_epochs - 1:
                            # and (self.gpu_id == 0 if self.args.ddp else True):
                            logging.info('Evaluating for epoch {}/{} ...'.format(epoch + 1, self.train_epochs))

                            current_state_student = self.trainer.trainer_eval(epoch, student_id, self.cp_used)

                            # optimize towards AUC or Accuracy
                            is_best = current_state_student[self.metric] > best_state_student[self.metric]
                            if is_best:
                                best_state_student.update(current_state_student)

                        # save model for later
                        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                              self.scheduler.state_dict(), epoch + 1, epoch_time,
                                              self.student_model.arch_info, self.student_model.hyper_info,
                                              best_state_student, is_best, self.save_dir, student_id)
                        # early stop check
                        if train_acc < last_train_acc:
                            epoch_no_improve += 1
                        else:
                            epoch_no_improve = 0

                        if self.early_stop and (epoch + 1) >= self.early_stop_epoch:
                            if best_state_student[self.metric] < self.early_stop_acc or \
                                    epoch_no_improve > self.early_stop_no_impr:
                                logging.info("Student {} EARLY STOP after {} epochs".format(student_id, epoch + 1))
                                break

                except torch.cuda.OutOfMemoryError:
                    logging.info("Student {} does NOT fit on GPU!".format(student_id))
                    torch.cuda.empty_cache()
                    oom_bool = True

                # save this stuff
                model_params_copy, hp_params_copy = copy.deepcopy(model_params), copy.deepcopy(hp_params)
                # for saving detach
                model_params = [x.cpu() for x in model_params]
                hp_params = [x.cpu() for x in hp_params]

                if not oom_bool:
                    # do not save oom architectures
                    rollouts.append([model_params_copy, hp_params_copy, best_state_student, student_id])
                    rollouts_save.append([model_params, hp_params, best_state_student, student_id])
                    action_list.append([student_id, actions_arch, actions_hp, best_state_student])

                # TODO put this into function
                # save rollout_save list
                with open('{}/rollouts.csv'.format(self.save_dir), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(rollouts_save)

                # save action list
                with open('{}/action_list.csv'.format(self.save_dir), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(action_list)

                # free gpu
                del model_params, hp_params
                del self.student_model, self.optimizer, self.trainer
                torch.cuda.empty_cache()
                gc.collect()
                student_id += 1

            if 0 <= self.warmup_epochs <= iteration:
                logging.info("Updating controller...")
                self.controller.update(rollouts, self.metric)

                # Determine validation accuracy of most likely student
                logging.info("Loading ARGMAX student...")
                model_params, hp_params, actions_arch_max, actions_hp_max = self.sample_student(argmax_id, argmax=True)

                # initialize argmax trainer
                self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                                       self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                                       self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                                       self.new_batch_size)

                # argmax student training loop
                argmax_time = 0
                if self.cp_used:
                    best_state_argmax = {'acc': 0, 'auc': 0, 'cm': 0, 'precision': 0, 'recall': 0, 'F1': 0, 'sens': 0,
                                         'spec': 0, 'ppv': 0, 'npv': 0, 'b_acc': 0}
                else:
                    best_state_argmax = {'acc': 0, 'acc5': 0, 'auc': 0, 'cm': 0}

                logging.info("Training ARGMAX student...")
                logging.info("Training ARGMAX student {} for {} epochs".format(argmax_id, self.argmax_epochs))
                try:
                    for epoch in range(self.argmax_epochs):
                        epoch_time, train_acc, train_loss = self.trainer.trainer_train(epoch, self.argmax_epochs,
                                                                                       argmax_id)
                        argmax_time += epoch_time
                        is_best = False
                        if (epoch + 1) % self.eval_interval == 0 or epoch >= self.argmax_epochs - 15:
                            # and (self.gpu_id == 0 if self.args.ddp else True):
                            # (self.gpu_id == 0 if self.args.ddp):
                            logging.info("Evaluating ARGMAX student quality in epoch {}/{}"
                                         .format(epoch + 1, self.argmax_epochs))
                            current_state_argmax = self.trainer.trainer_eval(epoch, argmax_id, self.cp_used)

                            # optimize towards AUC or Accuracy
                            is_best = current_state_argmax[self.metric] > best_state_argmax[self.metric]
                            if is_best:
                                best_state_argmax.update(current_state_argmax)

                        # save model for later
                        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                              self.scheduler.state_dict(), epoch + 1, epoch_time,
                                              self.student_model.arch_info, self.student_model.hyper_info,
                                              best_state_argmax, is_best, self.save_dir, argmax_id,
                                              argmax=True)

                except torch.cuda.OutOfMemoryError:
                    logging.info("ARGMAX student {} does not fit on GPU!".format(argmax_id))
                    torch.cuda.empty_cache()

                if self.args.ddp:
                    destroy_process_group()

                logging.info("ARGMAX student {}: Top1 {}, AUC {}, Training time: {}".format(
                    argmax_id, best_state_argmax['acc'], best_state_argmax['auc'], argmax_time))

                action_list.append([argmax_id, actions_arch_max, actions_hp_max, best_state_argmax])

                # save action list
                with open('{}/action_list.csv'.format(self.save_dir), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(action_list)

                if self.writer:
                    self.writer.add_scalar('Argmax_acc/student_{}'.format(argmax_id),
                                           best_state_argmax['acc'])
                    self.writer.add_scalar('Argmax_auc/student_{}'.format(argmax_id),
                                           best_state_argmax['auc'])
                    self.writer.add_scalar('Argmax_time/student_{}'.format(argmax_id), argmax_time)

                if best_state_argmax['acc'] >= 0.97:
                    logging.info('Controller converged!')
                    self.controller.converged = True

                del self.student_model, model_params, hp_params, self.trainer
                torch.cuda.empty_cache()
                argmax_id += 1

            else:
                logging.info("NOT updating controller!")

            average_quality = np.round(np.mean([r[2]['acc'] for r in rollouts]), 3)
            if self.writer:
                self.writer.add_scalar('Accuracy/val_avg', average_quality, iteration)
            logging.info("Average student quality over rollout is {}".format(average_quality))

            # save the controller stuff
            self.save_controller(iteration)

            if iteration >= self.args.max_iter:
                logging.info("Stopping after {} iterations".format(iteration))
                break

            # Update the policies
            if self.args.policy_deletion and (self.args.policy_updates <= iteration):
                logging.info("Deleting policies with threshold parameter: {}".format(self.args.policy_threshold))

                # find under performing policies -> get back indexes to delete them in dict
                hyper_space_update, arch_space_update = self.controller.delete_policies(self.arch_choices_copy,
                                                                                        self.hyper_choices_copy,
                                                                                        self.args.policy_threshold)

                self.hyper_choices = hyper_space_update
                self.hyper_computations = len(hyper_space_update)
                self.hyper_size = sum([len(x) for x in hyper_space_update.values()])

                self.arch_choices = arch_space_update
                self.arch_computations = len(arch_space_update)
                self.size_search = sum([len(x) for x in arch_space_update.values()])

                self.arch_names = []
                self.arch_values = []
                for items in self.arch_choices.items():
                    self.arch_names.append(items[0])
                    self.arch_values.append(items[1])

                self.hyper_names = []
                self.hyper_values = []
                for items in self.hyper_choices.items():
                    self.hyper_names.append(items[0])
                    self.hyper_values.append(items[1])

                logging.info("NEW Architecture Search Space is: {}".format(self.arch_choices))
                logging.info("NEW Hyperparameter Search Space is: {}".format(self.hyper_choices))
                logging.info("NEW Search Space size: {}".format(self.size_search + self.hyper_size))

            del rollouts
            iteration += 1

        # save final controller policy weights after convergence
        save_dir_conv = self.save_dir + '/controller_weights_converged/'
        self.controller.save_policies(save_dir_conv)
        logging.info("Converged after {} iterations!".format(iteration))

    def sample_student(self, student_id: int, argmax: bool):

        if argmax:
            model_params, hp_params = self.controller.policy_argmax(self.arch_computations, self.hyper_computations)
        else:
            model_params, hp_params = self.controller.sample(self.arch_computations, self.hyper_computations)

        # get indexes
        actions_arch_idx = [int(i) for i in model_params]
        actions_hyper_idx = [int(i) for i in hp_params]

        # generate student from sampled actions
        self.student_model = Student(actions_arch_idx, self.arch_choices, actions_hyper_idx, self.hyper_choices,
                                     student_id, self.args, **self.kwargs)

        logging.info("Student AP: {}".format(self.student_model.action_arch_dict))
        logging.info("Student HP: {}".format(self.student_model.actions_hyper_dict))
        # flops, params = thop.profile(self.student_model, inputs=torch.rand([1, 1] + self.data_shape),
        #                              verbose=False)
        # logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))

        # update hyperparameters for sampled student
        optimizer_fn = None
        for optimizer_class in self.optim_list:
            if optimizer_class.__name__.lower() == self.student_model.actions_hyper_dict['optimizers'].lower():
                optimizer_fn = optimizer_class
                break

        assert optimizer_fn is not None

        # fixed optimizer args from config
        optimizer_args = self.args.optimizer_args[self.student_model.actions_hyper_dict['optimizers']]
        # sampled optimizer args
        optimizer_args['lr'] = self.student_model.actions_hyper_dict['lr']
        optimizer_args['weight_decay'] = self.student_model.actions_hyper_dict['weight_decay']

        if optimizer_fn.__name__.lower() not in ['adam', 'adamw']:
            optimizer_args['momentum'] = self.student_model.actions_hyper_dict['momentum']

        self.new_batch_size = int(self.student_model.actions_hyper_dict['batch_size'])
        self.update_batch_size(self.new_batch_size)

        self.optimizer = optimizer_fn(params=self.student_model.parameters(), **optimizer_args)

        if self.args.lr_scheduler:
            # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
            self.scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                                      start_factor=self.args.sched_args.start_factor,
                                                                      total_iters=self.args.sched_args.warm_up)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=self.args.sched_args.step_lr,
                                                                  gamma=self.args.sched_args.gamma)
        else:
            self.scheduler = None
        logging.info('LR_Scheduler: {}'.format(self.scheduler.__class__.__name__))

        return model_params, hp_params, actions_arch_idx, actions_hyper_idx

    def save_controller(self, iteration: int) -> None:
        """
        Save histograms of controller policies
        """
        for idx, p in enumerate(self.controller.policies['archspace']):
            params = self.controller.policies['archspace'][p].state_dict()['params']
            params /= torch.sum(params)

            if self.writer:
                curr_policy = self.arch_names[idx]
                save_dict = {}
                for i in range(len(params)):
                    param_value = self.controller.arch_space.get(curr_policy)[i]
                    dict_name = '{}_{}'.format(curr_policy, param_value)
                    save_dict[dict_name] = params[i]

                self.writer.add_scalars('/Parameters/Arch/{}'.format(curr_policy), save_dict, iteration)

        for idx, p in enumerate(self.controller.policies['hpspace']):
            params = self.controller.policies['hpspace'][p].state_dict()['params']
            params /= torch.sum(params)

            if self.writer:
                curr_policy = self.hyper_names[idx]
                save_dict = {}
                for i in range(len(params)):
                    param_value = self.controller.hyper_space.get(curr_policy)[i]
                    dict_name = '{}_{}'.format(curr_policy, param_value)
                    save_dict[dict_name] = params[i]

                self.writer.add_scalars('/Parameters/Hyper/{}'.format(curr_policy), save_dict, iteration)

        logging.info("Saving controller policies...")
        save_dir = self.save_dir + self.args.controller_dir
        self.controller.save_policies(save_dir)
