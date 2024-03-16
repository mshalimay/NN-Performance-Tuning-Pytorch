
import os
import re
import torch
import pandas as pd
from load_and_evaluate import load_model, load_prepare_data, evaluate_model
import csv

def search_best_model(search_dir, model, batchsize, dataset, epochs):
    pattern = f"^{model}_{dataset}_batch={batchsize}_epoch={epochs}.*\\.pt$"
    best_accuracy = 0
    best_model_file = None
    for file in os.listdir(search_dir):
        if re.match(pattern, file):
            file_path = os.path.join(search_dir, file)
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

            if 'accuracy' in checkpoint:
                accuracy = checkpoint['accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_file = file

    print("Best model file: ", best_model_file)
    print("Best accuracy: ", best_accuracy)
    return best_model_file, best_accuracy

def collect_accuracy_data(search_dir, model, batchsize, dataset, epochs, evaluate:bool=False):
    pattern = f"^{model}_{dataset}_batch={batchsize}_epoch={epochs}.*\\.pt$"
    # Initialize a dictionary to store the data
    data = []
    data_test = load_prepare_data(dataset)[1]
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False)
    device = torch.device("cpu")
    for file in os.listdir(search_dir):
        if re.match(pattern, file):
            file_path = os.path.join(search_dir, file)
            try:
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            except:
                print(f"Error loading {file_path}")
                continue

            print(f"Processing {file_path}")

            if 'args' in checkpoint:
                args = checkpoint['args']
            else:
                print(f"No args in checkpoint. Skipping {file_path}")
                continue
                
            if evaluate:
                net = load_model(file_path)
                _, accuracy = evaluate_model(net, device, test_loader)
            else:
                accuracy = checkpoint['accuracy']

            
            # Assuming args has attributes like lr, sched (scheduler), and opt (optimizer)
            entry = {
                'lr': args.lr,
                'scheduler': args.sched,
                'optimizer': args.o,
                'accuracy': accuracy
            }
            data.append(entry)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # create table with metrics
    pivot_table = df.pivot_table(index=['optimizer', 'scheduler'], columns='lr', values='accuracy')

    #save the table to an Excel file
    pivot_table.to_excel(f'experiments_model={model}_dataset={dataset}_epoch={epochs}_batch={batchsize}.xlsx')

    return pivot_table


def collect_training_data(filename):
    highest_accuracy = 0
    time_to_highest_accuracy = 0
    total_time_elapsed = 0
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            accuracy = float(row['accuracy'])
            time_elapsed = float(row['time_elapsed'])

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            time_to_highest_accuracy = time_elapsed

        total_time_elapsed = time_elapsed
    return highest_accuracy, time_to_highest_accuracy, total_time_elapsed

saved_models_dir = "./saved_models"
train_log_dir = "./training_log"
model = "resmobile_4"
dataset = "cifar10"
search_dir = saved_models_dir
batchsize = 128
epochs = 100

collect_accuracy_data(saved_models_dir, model=model, dataset=dataset, batchsize=batchsize, epochs=epochs, evaluate=False)


# data = []
# for i in range(6):
#     model = f"resmobile_{i}"
#     filename, _ = search_best_model(search_dir, model=model, batchsize=batchsize, dataset=dataset, epochs=epochs)
#     train_log_filename = os.path.join(train_log_dir, filename.replace(".pt", ".csv"))
#     highest_accuracy, time_to_highest_accuracy, total_time_elapsed = collect_training_data(train_log_filename)
#     data.append([model, highest_accuracy, time_to_highest_accuracy, total_time_elapsed])

# filename =  "resnet_cifar10_batch=128_epoch=100_lr=1.0_opt=adadelta_sched=cosine_nest=1.pt"

# train_log_filename = os.path.join(train_log_dir, filename.replace(".pt", ".csv"))
# highest_accuracy, time_to_highest_accuracy, total_time_elapsed = collect_training_data(train_log_filename)
# data.append([model, highest_accuracy, time_to_highest_accuracy, total_time_elapsed])

# # Create a DataFrame
# df = pd.DataFrame(data, columns=['Model', 'Highest Accuracy', 'Time to Highest Accuracy', 'Total Time Elapsed'])

# # Export to Excel
# excel_filename = 'training_data_analysis.xlsx'
# df.to_excel(excel_filename, index=False)

# print(f"Data saved to {excel_filename}")

# print(collect_training_data("./training_log/resnet_cifar10_batch=128_epoch=100_lr=0.1_opt=sgd_sched=cosine_nest=1.csv"))