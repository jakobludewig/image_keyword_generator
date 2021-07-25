import click

from functions_build_keyword_database import *


@click.option(
    "--imagedir",
    prompt="Enter directory containing images to label",
    default=".",
    help="Directory containing the images to label.",
)
@click.option("--modelname", default="resnet152", help="Name of the model to use.")
@click.option(
    "--numtoplabels",
    default=10,
    help="The top n labels (by predicted probability) to extract for each image.",
)
@click.option(
    "--targetdbfile",
    default="keyword_database.pkl",
    help="Name of the file to write the extracted label information to.",
)
@click.command()
def build_keyword_database(imagedir, modelname, numtoplabels, targetdbfile):
    """Program to build a keyword database of the images in a folder structure that makes them searchable for certain objects."""
    click.echo("Labelling images in the following directory: {} ".format(imagedir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(modelname)

    dataset = build_dataset(imagedir=imagedir)
    click.echo("Number of images found in directory: {}".format(len(dataset)))

    filenames = [f for f, _ in dataset.imgs]

    dataloader = build_dataloader(dataset)

    probs = []
    with torch.no_grad():
        with click.progressbar(enumerate(dataloader), length=len(dataset) / 4) as bar:
            for _, (inputs, labels) in bar:
                inputs = inputs.to(device)

                outputs = model(inputs)
                probs.append(torch.nn.functional.softmax(outputs, dim=1))
    probs = torch.cat(probs)

    results_labels = extract_top_predicted_labels(
        probs=probs, numtoplabels=numtoplabels, filenames=filenames
    )

    write_results(results_labels, targetdbfile)

    click.echo("Done. Results available in file '{}'".format(targetdbfile))


if __name__ == "__main__":
    build_keyword_database()
