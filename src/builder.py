import collections

import numpy as np
import torch
import logging
import os

from src.initializer import Initializer
from src.model.trainer import Trainer
from src.model.student import Student
from src.utils import utils
import sys


class Builder(Initializer):
    """
    Simple Class for training and building a defined model
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.trainer = None
        self.student_model = None
        self.best_state = None
        self.argmax_epochs = self.args.argmax_epochs

        self.model_path = os.path.join(self.args.work_dir, self.args.retrain_model_path)
        logging.info(f"Loading model from {self.model_path}")

        if self.args.retrain_ensemble:
            self.model_path_2 = os.path.join(self.args.work_dir, self.args.retrain_model_path_2)
            logging.info(f"Loading second model from {self.model_path_2}")
        self.eval_interval = self.args.eval_interval

        self.cp_used = self.args.dataset in {'cp19', 'cp29'}
        self.metric = 'auc' if self.cp_used else 'acc'

    def retrain(self):
        """
        Public method to call.
        """
        # self.__retrain_student()

        if self.args.retrain_ensemble:
            preds_ensemble = []
            checkpoint_1 = self._load_weights(self.model_path)
            self.__build_trained_student(checkpoint_1)
            model_weights = collections.OrderedDict()
            # Copy contents of sub_dicts into model_weights
            model_weights.update(checkpoint_1['model']['input_stream'])
            model_weights.update(checkpoint_1['model']['main_stream'])
            model_weights.update(checkpoint_1['model']['classifier'])
            self.student_model.load_state_dict(model_weights)
            self.student_model.to(self.gpu_id)
            preds_1, labels_1, video_ids_1 = self.__test_ensemble()

            # softmax = np.zeros(preds.shape)
            # for i in range(preds.shape[0]):
            #     softmax[i, :] = np.exp(preds[i, :]) / np.sum(np.exp(preds[i, :]), axis=0)

            metrics_cp = self.trainer.evaluate_cp(preds_1, labels_1, video_ids_1, aggregate_binary=False,
                                                  subject=True)
            logging.info("Model 1 metrics:")
            logging.info(
                f'Accuracy: ({metrics_cp["acc"]:.2%}), AUC: {metrics_cp["auc"]:.4f}, '
                f'Precision: {metrics_cp["precision"]:.3f}, Recall: {metrics_cp["recall"]:.3f}, '
                f'F1: {metrics_cp["F1"]:.3f}, Sensitivity: {metrics_cp["sens"]:.3f}, Specificity: {metrics_cp["spec"]:.3f},'
                f' PPV: {metrics_cp["ppv"]:.3f}, NPV: {metrics_cp["npv"]:.3f}, Bal. Accuracy: {metrics_cp["b_acc"]:.3f} '
                f'CM: {metrics_cp["cm"]}'
            )


            preds_ensemble.append(preds_1)

            del model_weights, self.student_model

            checkpoint_2 = self._load_weights(self.model_path_2)
            self.__build_trained_student(checkpoint_2)
            model_weights = collections.OrderedDict()
            # Copy contents of sub_dicts into model_weights
            model_weights.update(checkpoint_2['model']['input_stream'])
            model_weights.update(checkpoint_2['model']['main_stream'])
            model_weights.update(checkpoint_2['model']['classifier'])
            self.student_model.load_state_dict(model_weights)
            self.student_model.to(self.gpu_id)
            preds_2, labels_2, video_ids_2 = self.__test_ensemble()

            metrics_cp = self.trainer.evaluate_cp(preds_2, labels_2, video_ids_2, aggregate_binary=False,
                                                  subject=True)
            logging.info("Model 2 metrics:")
            logging.info(
                f'Accuracy: ({metrics_cp["acc"]:.2%}), AUC: {metrics_cp["auc"]:.4f}, '
                f'Precision: {metrics_cp["precision"]:.3f}, Recall: {metrics_cp["recall"]:.3f}, '
                f'F1: {metrics_cp["F1"]:.3f}, Sensitivity: {metrics_cp["sens"]:.3f}, Specificity: {metrics_cp["spec"]:.3f},'
                f' PPV: {metrics_cp["ppv"]:.3f}, NPV: {metrics_cp["npv"]:.3f}, Bal. Accuracy: {metrics_cp["b_acc"]:.3f} '
                f'CM: {metrics_cp["cm"]}'
            )

            # Define the possible weightings you want to test
            weight_combinations = [
                ([1.0, 0.0], [1.0, 0.0]),  # Full influence of Model 1 for both classes
                ([0.0, 1.0], [0.0, 1.0]),  # Full influence of Model 2 for both classes

                ([1.0, 0.0], [0.0, 1.0]),  # Model 1 dominates class 0, Model 2 dominates class 1
                ([0.6, 0.4], [0.6, 0.8]),  # Stronger influence for Model 1 in class 0, slightly more Model 2 in class 1
                ([0.5, 0.3], [0.5, 0.7]),  # Model 1 more influence in both classes, but Model 2 stronger in class 1
                ([0.6, 0.4], [0.4, 0.6]),  # Slightly more weight for Model 1 in class 0, balanced in class 1
                ([0.7, 0.3], [0.3, 0.7]),  # Model 1 has much more influence in class 0, Model 2 dominates class 1
                ([0.8, 0.2], [0.2, 0.8]),  # Strong emphasis on Model 1 in class 0, strong Model 2 for class 1
                ([0.5, 0.5], [0.5, 0.5]),  # Equal weights for both models in both classes
                ([0.0, 1.0], [1.0, 0.0]),  # Model 2 dominates class 0, Model 1 dominates class 1

                # Extended combinations
                ([0.9, 0.1], [0.1, 0.9]),  # Strong preference for Model 1 in class 0, strong for Model 2 in class 1
                ([0.85, 0.15], [0.15, 0.85]),  # More weight for Model 1 in class 0, stronger Model 2 in class 1
                ([0.75, 0.25], [0.25, 0.75]),  # Balanced towards Model 1 in class 0, Model 2 in class 1
                ([0.6, 0.4], [0.2, 0.8]),  # Model 1 stronger in class 0, much more influence for Model 2 in class 1
                ([0.4, 0.6], [0.6, 0.4]),  # More weight for Model 2 in class 0, more weight for Model 1 in class 1
                ([0.7, 0.3], [0.5, 0.5]),  # Strong influence for Model 1 in class 0, equal in class 1
                ([0.3, 0.7], [0.5, 0.5]),  # More influence for Model 2 in class 0, equal in class 1
                ([0.8, 0.2], [0.6, 0.4]),  # Model 1 stronger for both classes, but heavier for class 0
                ([0.5, 0.5], [0.4, 0.6]),  # Equal weight for class 0, stronger Model 2 in class 1
                ([0.4, 0.6], [0.7, 0.3]),  # Model 2 dominates class 0, Model 1 dominates class 1
                ([0.9, 0.1], [0.9, 0.1]),  # High influence of Model 1 for both classes
                ([0.1, 0.9], [0.1, 0.9]),  # High influence of Model 2 for both classes
            ]

            for weights in weight_combinations:
                weight_class_1, weight_class_0 = weights
                logging.info(
                    f'Testing weight combination: Class 0 weights {weight_class_0}, Class 1 weights {weight_class_1}')

                weighted_preds = np.zeros_like(preds_1)

                # Loop over each sample
                for i in range(preds_1.shape[0]):
                    # Apply different weights for each class
                    weighted_preds[i, 0] = (weight_class_0[0] * preds_1[i, 0] + weight_class_0[1] * preds_2[i, 0]) / 2  # Class 0
                    weighted_preds[i, 1] = (weight_class_1[0] * preds_1[i, 1] + weight_class_1[1] * preds_2[i, 1]) / 2  # Class 1

                # Evaluate the ensemble predictions using your evaluation function
                metrics_cp = self.trainer.evaluate_cp(weighted_preds, labels_1, video_ids_1, aggregate_binary=False,
                                                      subject=True)

                # Log the evaluation metrics for the current weighting
                logging.info("Model ensemble evaluation metrics:")
                logging.info(
                    f'Weights Class 0: {weight_class_0}, Weights Class 1: {weight_class_1} -> '
                    f'Accuracy: {metrics_cp["acc"]:.2%}, AUC: {metrics_cp["auc"]:.4f}, '
                    f'Precision: {metrics_cp["precision"]:.3f}, Recall: {metrics_cp["recall"]:.3f}, '
                    f'F1: {metrics_cp["F1"]:.3f}, Sensitivity: {metrics_cp["sens"]:.3f}, Specificity: {metrics_cp["spec"]:.3f}, '
                    f'PPV: {metrics_cp["ppv"]:.3f}, NPV: {metrics_cp["npv"]:.3f}, Balanced Accuracy: {metrics_cp["b_acc"]:.3f}, '
                    f'Confusion Matrix: {metrics_cp["cm"]}'
                )

        else:
            checkpoint = self._load_weights(self.model_path)
            self.__build_trained_student(checkpoint)
            model_weights = collections.OrderedDict()
            # Copy contents of sub_dicts into model_weights
            model_weights.update(checkpoint['model']['input_stream'])
            model_weights.update(checkpoint['model']['main_stream'])
            model_weights.update(checkpoint['model']['classifier'])
            self.student_model.load_state_dict(model_weights)
            self.student_model.to(self.gpu_id)
            # self.__retrain_student()
            self.__test(ensemble=False)

    def __test_ensemble(self):
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.test_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        preds, labels, video_ids = self.trainer.trainer_eval(0, 1, self.cp_used, True)
        return preds, labels, video_ids


    def __test(self, ensemble=False):
        # initialize argmax trainer
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.test_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        argmax_id = 9999
        epoch = 0
        epoch_time = 0
        best_state_argmax = {'acc': 0, 'auc': 0, 'cm': 0, 'precision': 0, 'recall': 0, 'F1': 0, 'sens': 0,
                                 'spec': 0, 'ppv': 0, 'npv': 0, 'b_acc': 0}

        input = torch.rand(1, *tuple(self.test_loader.dataset.shape))
        macs, params = utils.compute_macs_and_params(self.student_model, input.to(self.gpu_id))
        logging.info(f"Model has: {macs} MACs and {params} params")

        current_state_argmax = self.trainer.trainer_eval(0, argmax_id, self.cp_used)
        is_best = current_state_argmax[self.metric] > best_state_argmax[self.metric]
        if is_best:
            best_state_argmax.update(current_state_argmax)

        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                              self.scheduler.state_dict(), epoch + 1, epoch_time,
                              self.student_model.arch_info, self.student_model.hyper_info,
                              best_state_argmax, is_best, self.save_dir, argmax_id, argmax=True)

    def _load_weights(self, path: str):
        """
        Load weights into the model from the specified file.
        """
        try:
            # load trained student or argmax architecture and check if all streams are saved
            checkpoint = torch.load(path, map_location=self.gpu_id)
            model_state = ['input_stream', 'main_stream', 'classifier']
            # check if all streams are stored in the
            if not all(state_name in checkpoint['model'] for state_name in model_state):
                logging.info("Loaded checkpoint but not usable with this version!")
                logging.info("Please retrain the model!")
                sys.exit()
            return checkpoint
        except FileNotFoundError:
            logging.error("File not found at: {}".format(path))
            logging.info("Please check the path an try again!")
            sys.exit()

    def __build_trained_student(self, checkpoint, student_id=0) -> None:
        """
        Build the student for the xai method.
        May have to retrain the model due to old bug...
        @param student_id:
        @return:
        """
        try:
            # check if hyperparameters are stored in model otherwise get them from the config file
            actions_hyper = checkpoint['actions_hyper']
            assert len(actions_hyper) != 0
        except (AssertionError, KeyError):
            logging.info("Hyperparameter values not provided in checkpoint file!")
            logging.info("Trying to load them from config file...")
            if len(self.args.xai_hyper_param) != 0:
                actions_hyper = list(self.args.xai_hyper_param)
            else:
                logging.info("Hyperparameter values not valid!")
                logging.info("Please provide the hyperparameter values manually in the config file!")
                sys.exit()

        # get indices for student class -> should be a dict from model
        actions_arch_dict = checkpoint['actions']

        self.student_model = Student(actions_arch_dict, self.arch_choices, actions_hyper, self.hyper_choices,
                                     student_id, self.args, **self.kwargs)

        logging.info("Student AP: {}".format(self.student_model.action_arch_dict))
        logging.info("Student HP: {}".format(self.student_model.actions_hyper_dict))
        # flops, params = thop.profile(deepcopy(self.student_model), inputs=torch.rand([1, 1] + self.data_shape),
        #                             verbose=False)
        # logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))

        # update hyperparameters for sampled student
        optimizer_fn = None
        for optimizer_class in self.optim_list:
            if optimizer_class.__name__.lower() == self.student_model.hyper_info['optimizers'].lower():
                optimizer_fn = optimizer_class
                break

        assert optimizer_fn is not None

        # fixed optimizer args from config
        optimizer_args = self.args.optimizer_args[self.student_model.hyper_info['optimizers']]
        # sampled optimizer args
        optimizer_args['lr'] = self.student_model.actions_hyper_dict['lr']
        optimizer_args['weight_decay'] = self.student_model.hyper_info['weight_decay']

        if optimizer_fn.__name__.lower() not in ['adam', 'adamw']:
            optimizer_args['momentum'] = self.student_model.hyper_info['momentum']

        self.new_batch_size = int(self.student_model.hyper_info['batch_size'])
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
        logging.info('Learning rate scheduler: {}'.format(self.scheduler.__class__.__name__))
        logging.info("Done with initializing student skeleton...")

    def __retrain_student(self):
        """
        Simple retrain function.
        """
        # initialize argmax trainer
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        argmax_time = 0
        argmax_id = 2
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
                                      best_state_argmax, is_best, self.save_dir, argmax_id, argmax=False)
        except torch.cuda.OutOfMemoryError:
            logging.info("ARGMAX student {} does not fit on GPU!".format(argmax_id))
            torch.cuda.empty_cache()

        logging.info("ARGMAX student {}: Top1 {}, AUC {}, Training time: {}".format(
            argmax_id, best_state_argmax['acc'], best_state_argmax['auc'], argmax_time))
        logging.info("Done with retraining...")



