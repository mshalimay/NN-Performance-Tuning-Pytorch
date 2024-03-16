import os
max_epoch = 100

datasets = ['cifar10']
models = ['resnet', 'resmobile', 'simplenet']
resmob_options = [0, 1, 2, 3, 4, 5]
batchsizes = [64, 128, 256, 512]
lrs = [1.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
optimizers = ['sgd', 'adadelta', 'adam']
schedulers = ['step', 'plateau', 'cosine', 'cosine_r', 'cyclic', 'none']
nesterov = [1, 0]

for resmob_option in resmob_options:
    for dataset in datasets:
        for model in models:
            for batchsize in batchsizes:
                for lr in lrs:
                    for optimizer in optimizers:
                        for scheduler in schedulers:
                            for nest in nesterov:
                                model_path = f"saved_models/{model}_{dataset}_batch={batchsize}_epoch={max_epoch}_lr={lr}_opt={optimizer}_sched={scheduler}_nest={nest}.pt"
                                if os.path.exists(model_path):
                                    print(f"{model_path} already exists. Skipping...")
                                    continue
                                if optimizer != 'sgd' and nest==1:
                                    # no need for nesterov if not using SGD
                                    continue
                                # send cmd command to terminal
                                cmd = (f"python train.py --model {model} --dataset {dataset} "
                                    f"--epochs {max_epoch} --batch-size {batchsize} --lr {lr} --o {optimizer} "
                                    f"--sched {scheduler} --nest {nest} --save-model --log-train --log-interval -1")

                                print("Training model: ", model_path)
                                os.system(cmd)