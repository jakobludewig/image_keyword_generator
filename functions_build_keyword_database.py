import pandas as pd

from typing import List

import torch
from torchvision import models, transforms, datasets


data_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(modelname: str = "resnet152"):
    """ Loads a pretrained model from the PyTorch model zoo (currently only 'resnet152' can be used)

    Args:
        modelname (str, optional): Name of model to load. Defaults to "resnet152".

    Returns:
        Pretrained model object
    """

    if modelname == "resnet152":
        model = models.resnet152(pretrained=True)
        model = model.eval()
        return model
    else:
        return None


def build_dataset(
    imagedir: str, transform: transforms.Compose = data_transform
) -> datasets.ImageFolder:
    """ Creates a datasets.ImageFolder object to access the images and apply transformations when loading them

    Args:
        imagedir (str): Directory structure containing images
        transform (transforms.Compose, optional): Transformations to apply to images when loading them. Defaults to data_transform.

    Returns:
        datasets.ImageFolder: [description]
    """
    return datasets.ImageFolder(imagedir, transform)


def build_dataloader(dataset: datasets.ImageFolder) -> torch.utils.data.DataLoader:
    """ Create DataLoader to read in the images in batches.

    Args:
        dataset (datasets.ImageFolder): datasets.ImageFolder providing access to the images

    Returns:
        torch.utils.data.DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4
    )


def extract_top_predicted_labels(
    probs: torch.Tensor, numtoplabels: int, filenames: List[str]
) -> pd.DataFrame:
    """[summary]

    Args:
        probs (torch.Tensor): Tensor containing the predictions from the model
        numtoplabels (int): Maximum number of labels to extract for each image
        filenames (List[str]): Filenames of the images

    Returns:
        pd.DataFrame: DataFrame containing filenames of images along with top labels predicted by the model
    """

    top_predictions = torch.topk(probs, dim=1, k=numtoplabels)

    column_names = [i for i in list(range(1, numtoplabels + 1))]

    top_labels = (
        pd.DataFrame(
            top_predictions.indices.numpy(), columns=column_names, index=filenames
        )
        .melt(var_name="pred_rank", value_name="label", ignore_index=False)
        .reset_index(drop=False)
        .merge(
            pd.DataFrame(
                top_predictions.values.numpy(), columns=column_names, index=filenames
            )
            .melt(var_name="pred_rank", value_name="prob", ignore_index=False)
            .reset_index(drop=False),
            on=["pred_rank", "index"],
        )
        .rename(columns={"index": "filename"})
    )
    top_labels = top_labels.sort_values(["filename", "pred_rank"]).reset_index(
        drop=True
    )
    return top_labels


def write_results(results_labels: pd.DataFrame, targetdbfile: str) -> None:
    """ Write keywords database to file

    Args:
        results_labels (pd.DataFrame): DataFrame containing filenames of images along with top labels predicted by the model
        targetdbfile (str): Filename to write the DataFrame to as a pickle
    """
    results_labels.to_pickle(targetdbfile)
