import time

import torch
from metrics.distance_metrics import DistanceEvaluation
from models.classifier import Classifier

import wandb
from utils.training_config_parser import TrainingConfigParser
import matplotlib


def load_model(run_path,
               model_path=None,
               config=None,
               architecture=None,
               num_classes=None,
               replace=True):

    # Get file path at wandb if not set
    if model_path is None:
        api = wandb.Api(timeout=60)
        run = api.run(run_path)
        model_path = run.config["model_path"]
        #model_path = '/root/Plug-and-Play-Attacks/results/resnest101_20220310_041030/Classifier_0.9648_no_val.pth'
        architecture = run.config['Architecture']

    # Create model
    if num_classes is None:
        num_classes = run.config["num_classes"]

    if config:
        model = config.create_model()
    elif architecture is None:
        architecture = model_path.split('/')[-1].split('_')[0]

    model = Classifier(num_classes, in_channels=3, architecture=architecture)
    checkpoint = torch.load(model_path,map_location='cpu')['model_state_dict']
    model.load_state_dict(checkpoint)
                
    # Load weights from wandb
    #file_model = wandb.restore(model_path,
                               #run_path=run_path,
                               #root='./weights',
                               #replace=replace)

    # Load weights from local file
    #model.load_state_dict(
        #torch.load(file_model.name, map_location='cpu')['model_state_dict'])

    model.wandb_name = run.name

    return model

def load_model_eval(run_path,
               model_path=None,
               config=None,
               architecture=None,
               num_classes=None,
               replace=True):

    # Get file path at wandb if not set
    if model_path is None:
        api = wandb.Api(timeout=60)
        run = api.run(run_path)
        model_path = run.config["model_path"]
        architecture = run.config['Architecture']

    # Create model
    if num_classes is None:
        num_classes = run.config["num_classes"]

    if config:
        model = config.create_model()
    elif architecture is None:
        architecture = model_path.split('/')[-1].split('_')[0]

    model = Classifier(num_classes, in_channels=3, architecture=architecture)
    checkpoint = torch.load(model_path,map_location='cpu')['model_state_dict']
    model.load_state_dict(checkpoint)
                
    # Load weights from wandb
    #file_model = wandb.restore(model_path,
                               #run_path=run_path,
                               #root='./weights',
                               #replace=replace)

    # Load weights from local file
    #model.load_state_dict(
        #torch.load(file_model.name, map_location='cpu')['model_state_dict'])

    model.wandb_name = run.name

    return model

def load_config(run_path, config_name):
    config_file = wandb.restore(config_name,
                                run_path=run_path,
                                root='./configs',
                                replace=True)
    with open(config_file.name, "r") as config:
        config = TrainingConfigParser(config)
    return config


def log_attack_progress(loss,
                        target_loss,
                        discriminator_loss,
                        discriminator_weight,
                        mean_conf,
                        lr,
                        imgs=None,
                        captions=None):
    if imgs is not None:
        imgs = [
            wandb.Image(img.permute(1, 2, 0).numpy(), caption=caption)
            for img, caption in zip(imgs, captions)
        ]
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr,
            'samples': imgs
        })
    else:
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr
        })


def init_wandb_logging(optimizer, target_model_name, config, args):
    lr = optimizer.param_groups[0]['lr']
    optimizer_name = type(optimizer).__name__
    if not 'name' in config.wandb['wandb_init_args']:
        config.wandb['wandb_init_args']['name'] = f'{optimizer_name}_{lr}_{target_model_name}'
    wandb_config = config.create_wandb_config()
    run = wandb.init(config=wandb_config,
                     **config.wandb['wandb_init_args'])
    wandb.save(args.config)
    return run


def intermediate_wandb_logging(optimizer, targets, confidences, loss,
                               target_loss, discriminator_loss,
                               discriminator_weight, mean_conf, imgs, idx2cls):
    lr = optimizer.param_groups[0]['lr']
    target_classes = [idx2cls[idx.item()] for idx in targets.cpu()]
    conf_list = [conf.item() for conf in confidences]
    if imgs is not None:
        img_captions = [
            f'{target} ({conf:.4f})'
            for target, conf in zip(target_classes, conf_list)
        ]
        log_attack_progress(loss,
                            target_loss,
                            discriminator_loss,
                            discriminator_weight,
                            mean_conf,
                            lr,
                            imgs,
                            captions=img_captions)
    else:
        log_attack_progress(loss, target_loss, discriminator_loss,
                            discriminator_weight, mean_conf, lr)


def log_nearest_neighbors(imgs, targets, eval_model, model_name, dataset, img_size, seed, device):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(
        eval_model, None, img_size, None, dataset, seed, device)
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)
    closest_samples = [
        wandb.Image(img.permute(1, 2, 0).cpu().numpy(),
                    caption=f'distance={d:.4f}')
        for img, d in zip(closest_samples, distances)
    ]
    wandb.log({f'closest_samples {model_name}': closest_samples})

def log_final_images(imgs, predictions, max_confidences, target_confidences, idx2cls):
    wand_imgs = [
        wandb.Image(
            img.permute(1, 2, 0).numpy(),
            caption=f'pred={idx2cls[pred.item()]} ({max_conf:.2f}), target_conf={target_conf:.2f}'
        ) for img, pred, max_conf, target_conf in zip(
            imgs.cpu(), predictions, max_confidences, target_confidences)
    ]
    wandb.log({'final_images': wand_imgs})
    


def final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1, acc_top5,
                        avg_dist_facenet, avg_dist_eval, fid_score, precision, recall, density, coverage):
    wandb.save('attacks/gradient_based.py')
    wandb.run.summary['correct_avg_conf'] = avg_correct_conf
    wandb.run.summary['total_avg_conf'] = avg_total_conf
    wandb.run.summary['evaluation_acc@1'] = acc_top1
    wandb.run.summary['evaluation_acc@5'] = acc_top5
    wandb.run.summary['avg_dist_facenet'] = avg_dist_facenet
    wandb.run.summary['avg_dist_evaluation'] = avg_dist_eval
    wandb.run.summary['fid_score'] = fid_score
    wandb.run.summary['precision'] = precision
    wandb.run.summary['recall'] = recall
    wandb.run.summary['density'] = density
    wandb.run.summary['coverage'] = coverage

    wandb.finish()
