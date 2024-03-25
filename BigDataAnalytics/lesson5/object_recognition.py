# Best done in a Jupyter Notebook

# import torch
# print(torch.cuda.is_available())
#
# !pip install detecto
#
# from detecto import core, utils, visualize
# from detecto.visualize import show_labeled_image, plot_prediction_grid
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
# custom_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(900),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.ColorJitter(saturation=0.2),
#     transforms.ToTensor(),
#     utils.normalize_transform(),
# ])
#
# Train_dataset = core.Dataset('TrainJaguar/',transform=custom_transforms) #L1
# Test_dataset  = core.Dataset('TestJaguar/')#L2
# loader        = core.DataLoader(Train_dataset, batch_size=2, shuffle=False)
#
# # These labels correspond to the labels assigned in makesense.ai
# model  = core.Model(['eye', 'mouth'])
#
# # Fit the model.
# losses = model.fit(loader, Test_dataset, epochs=50,
#                    lr_step_size=5, learning_rate=0.001, verbose=True)
#
# model.save('model_weights.pth')
# model = core.Model.load('model_weights.pth', ['eye', 'mouth'])
#
# image                 = utils.read_image('TestJaguar/1122614.jpg')
# predictions           = model.predict(image)
# labels, boxes, scores = predictions
# show_labeled_image(image, boxes, labels)
#
# thresh=0.5
# filtered_indices=np.where(scores>thresh)
# filtered_scores  = scores[filtered_indices]
# filtered_boxes   = boxes[filtered_indices]
# num_list         = filtered_indices[0].tolist()
# filtered_labels  = [labels[i] for i in num_list]
# show_labeled_image(image, filtered_boxes, filtered_labels)
