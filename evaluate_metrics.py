import torch_fidelity, os
from data import create_inria_dataloaders
def compute_metrics():
    CROP_SIZE = 64
    #wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(generator, CROP_SIZE, 'normal', 0)
    train_ds, test_ds=create_inria_dataloaders(batch_size=128,image_size=CROP_SIZE, return_dataset=True)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=train_ds,
        input2="results/inriafull/samples", 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=False, 
        verbose=False,
    )
    print(metrics_dict)
    breakpoint()

# fidelity --gpu 0 --isc --fid --kid --input1 maps/train --input2 outs/maps64_40k # better to divide gt input and out into folders
# fidelity --isc --fid --kid --input1 "maps/train" --input2 "outs/maps64_40k"

if __name__ == "__main__":
    compute_metrics()