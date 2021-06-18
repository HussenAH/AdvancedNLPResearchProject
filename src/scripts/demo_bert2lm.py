#import ..src.BERT2LM.utils as util
import sys
#sys.path.insert(0, '../src')
from tabulate import tabulate
import argparse
from src.BERT2LM import utils

MODEL_DIR_MAP = {
    "BERT-large-augmented" : "",
    "BERT-large-baseline": "../BERT_WSD/model/bert_large-batch_size=128-lr=2e-5-max_gloss=6"
}

def get_model_dir(type):
    if type in MODEL_DIR_MAP:
        return MODEL_DIR_MAP.get(type)
    raise EnvironmentError("Error no file named {} found in directory {}".format(
        type, MODEL_DIR_MAP))

def main(model_type):

    _model_dir = get_model_dir(model_type)

    # Load fine-tuned model and vocabulary
    model, tokenizer = utils.load_bert_wsd_model(_model_dir)

    while True:
        try:
            sentence = input("\nEnter a sentence with an ambiguous word surrounded by [TGT] tokens\n> ")
            predictions = utils.get_predictions(model, tokenizer, sentence)
            if predictions:
                print("\nPredictions:")
                print(tabulate(
                    [[f"{i+1}.", key, gloss, f"{score:.5f}"] for i, (key, gloss, score) in enumerate(predictions)],
                    headers=["No.", "Sense key", "Definition", "Score"])
                )
                # for i, (sense_key, definition, score) in enumerate(predictions):
                #     # print(f"  {i + 1:>3}. sense key: {sense_key:<15} score: {score:<8.5f} definition: {definition}")
        except  Exception as e:
            print(str(e))


main("BERT-large-baseline")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "model_type",
        default=None,
        type=str,
        help="Directory of pre-trained model."
    )
    args = parser.parse_args()
    main(args.model_type)
