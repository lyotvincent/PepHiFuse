"""
Language modeling evaluation script

"""
import json
import logging
import math
import os
import sys
from time import time

import pandas as pd
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    set_seed,
)

from terminator.args import CustomTrainingArguments, EvalArguments
from terminator.collators import (
    ConditionalGenerationEvaluationCollator,
    PropertyCollator,
)
from terminator.datasets import get_dataset
from terminator.evaluator import Evaluator
from terminator.property_predictors import PREDICT_FACTORY
from terminator.tokenization import ExpressionBertTokenizer
from terminator.trainer import get_trainer_dict, SelfDefined_Config, SelfDefined_Model
from terminator.utils import (
    disable_rdkit_logging,
    find_safe_path,
    get_latest_checkpoint,
    get_equispaced_ranges,
)

logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((CustomTrainingArguments, EvalArguments))
    training_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    with open(eval_args.param_path, "r") as f:
        eval_params = json.load(f)

    param_filename = eval_args.param_path.split("/")[-1].split(".json")[0]

    # Wrap into args to be safe
    eval_args.__dict__.update(eval_params)

    # NOTE: Results will be stored in model folder
    model_dir = training_args.output_dir
    if "checkpoint" not in model_dir:
        model_dir = get_latest_checkpoint(
            model_dir, must_contain=eval_params.get("checkpoint-str", "best")
        )
    logger.warning(f"Entering in {model_dir} directory.")

    config_name = os.path.join(model_dir, "config.json")
    with open(config_name, "r") as f:
        model_params = json.load(f)

    selfdefined_config = SelfDefined_Config.from_pretrained(config_name)

    tokenizer = ExpressionBertTokenizer.from_pretrained(f'{model_dir}/vocab.txt')
    sep = tokenizer.expression_separator

    model = SelfDefined_Model.from_pretrained(model_dir, config=selfdefined_config, tokenizer=tokenizer)
    logger.info(f"Model restored from {model_dir}")

    assert model.config.transformer_config["vocab_size"] == len(tokenizer)
    model.transformer_resize_token_embeddings(len(tokenizer))

    assert eval_params["block_size"] > 0

    # Get datasets
    eval_dataset = get_dataset(
        eval_args.eval_file,
        block_size=eval_params["block_size"],
        tokenizer=tokenizer,
        line_by_line=eval_params.get("line_by_line", True),
    )
    for i in range(len(eval_dataset)):
        eval_dataset.examples[i] = tokenizer.add_padding_tokens(eval_dataset.examples[i], eval_params["block_size"])

    logger.info(f"Dataset size {len(eval_dataset)}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")

    plm_prob = eval_params["plm_probability"]
    perplexity_plm_prob = eval_params.get("perplexity_plm_prob", 0.2)
    # NOTE: This collator does not provide an attention mask (unlike the refined training
    # collators which prevent attention on padding), however, the model will largely
    # ignore the paddings.
    vanilla_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=perplexity_plm_prob,
        max_span_length=eval_params["max_span_length"],
    )

    custom_trainer_params = get_trainer_dict(model_params['transformer_config'])

    # Initialize our Evaluator
    evaluator = Evaluator(
        model=model,
        args=training_args,
        eval_params=eval_params,
        data_collator=vanilla_collator,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        prediction_loss_only=False,
        **custom_trainer_params,
    )

    # Evaluation
    result_dir = os.path.join(model_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    eval_filename = eval_args.eval_file.split("/")[-1].split(".")[0]
    # logger.info("*** Evaluate perplexity ***")

    # with open(eval_args.eval_file, "r") as f:
    #     prefix = sep.join(f.readline().split(sep)[:-1]) + sep

    # Set seed
    if eval_params.get("set_seed", True):
        set_seed(eval_params.get("seed", 42))

    # eval_output = evaluator.evaluate()
    # perplexity = math.exp(eval_output["eval_loss"])
    # results = {"perplexity": perplexity}
    # path = os.path.join(
    #     result_dir, f"{eval_filename}_perplexity_plm_{perplexity_plm_prob}.txt"
    # )

    # with open(find_safe_path(path), "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(results.keys()):
    #         logger.info("  %s = %s", key, str(results[key]))
    #         writer.write("%s = %s\n" % (key, str(results[key])))

    disable_rdkit_logging()
    property_results = []
    properties = eval_params["property_tokens"]
    orders = eval_params.get("property_token_masking_order", None)
    tokens_to_mask = eval_params.get("property_tokens_to_mask", None)
    conditioning_ranges = eval_params.get(
        "conditioning_range",
        get_equispaced_ranges(
            eval_args.eval_file,
            properties,
            precisions=eval_params.get("property_precisions", [2] * len(properties)),
        ),
    )
    logger.info(f"Conditioning range is {conditioning_ranges}")

    # If the token masking orders is not specified we just evaluate all properties together
    if not orders:
        property_collator = PropertyCollator(
            tokenizer=tokenizer,
            property_tokens=properties,
            num_tokens_to_mask=tokens_to_mask,
            mask_token_order=orders,
        )
        ps, rs = evaluator.multi_property_prediction(
            property_collator,
            save_path=os.path.join(result_dir, eval_filename),
            rmse_factor=eval_params.get("rmse_factor", 1),
        )
    else:

        for prop, order, mask in zip(properties, orders, tokens_to_mask):
            logger.info(f"*** Evaluate property {prop} ***")

            # The order of mask is determined by `property_token_masking_order`.
            property_collator = PropertyCollator(
                tokenizer=tokenizer,
                property_tokens=[prop],
                num_tokens_to_mask=[mask],
                mask_token_order=[order],
            )
            print(f"Masking {mask} tokens in order {order}")
            # if mask < len(total_tokens): only get the mask tokens
            ps, rs, ss, _metrics, ars = evaluator.property_prediction(
                property_collator,
                save_path=os.path.join(
                    result_dir, f"{prop[1:-1]}_{eval_filename}_mask_{mask}.csv"
                ),
                rmse_factor=eval_params.get("rmse_factor", 1),
            )
            search_methods = ["Greedy", "Sampling"]
            median_ars = np.zeros((len(search_methods)))
            for i, ar in enumerate(ars):
                median_ars[i] = np.median(ar)
            for p, r, s, mar, n in zip(ps, rs, ss, median_ars, search_methods):
                prop_res_dict = {
                    "prop": prop[1:-1],
                    "pearson": p,
                    "spearman": s,
                    "rmse": r,
                    "mar": mar,
                    "search": n,
                    "num_masked": mask,
                }
                property_results.append(prop_res_dict)

            pd.DataFrame(property_results).to_csv(
                os.path.join(result_dir, f"property_prediction_{eval_filename}.csv")
            )
    return
    for prop, cr in zip(properties, conditioning_ranges):
        logger.info(f"Evaluating conditional generation for {prop} with {cr}")
        conditional_generation_collator = ConditionalGenerationEvaluationCollator(
            tokenizer=tokenizer,
            property_token=prop,
            conditioning_range=cr,
            plm_probability=plm_prob,
            max_span_length=eval_params["max_span_length"],
            entity_to_mask=eval_params.get("entity_to_mask", None),
            entity_separator_token=eval_params.get("entity_separator_token", None),
        )

        # Retrieve the property prediction function from dictionary
        if prop[1:-1] in PREDICT_FACTORY.keys():
            evaluate_fn = PREDICT_FACTORY[prop[1:-1]]
            logger.info(f"Found property predictor for {prop}")
            property_collator = None
        else:
            # If unavailable property is predicted
            evaluate_fn = None

            if orders:
                # In single property prediction mode we just mask the property
                property_collator = PropertyCollator(
                    tokenizer=tokenizer,
                    property_tokens=[prop],
                    num_tokens_to_mask=[-1],
                    mask_token_order=None,
                )
            else:
                # in this case, we use the property predictor from above where all tokens are masked
                pass

            logger.info(
                f"No property predictor for {prop}, using model itself for evaluation"
            )

        evaluator.conditional_generation(
            conditional_generation_collator,
            save_path=os.path.join(
                result_dir,
                f"{prop[1:-1]}_conditional_generation_{param_filename}_{eval_filename}.csv",
            ),
            passed_eval_fn=evaluate_fn,
            property_collator=property_collator,
            denormalize_params=eval_params.get("denormalize", {}).get(prop, None),
            # prefix=prefix,
        )

    print("Done, shutting down.")


if __name__ == "__main__":
    main()
