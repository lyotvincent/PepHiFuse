"""
Generate rt data.

python scripts/generate_rt_data.py rt_data/SAL00031/SAL00031_train_eval.tsv rt_data/SAL00031/train.txt --sample_ratio 0.8

or

python scripts/generate_rt_data.py rt_data/SAL00031/SAL00031_test.tsv rt_data/SAL00031/test.txt

NOTE: assert all values < 1000, > -1000 and are in minutes.
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_filepath", type=str, help="path to the .tsv file.")
parser.add_argument("output_filepath", type=str, help="output where to store the data.")
parser.add_argument("--sample_ratio", type=float, default=1.0, help="proportion of output data to input data, the rest is set as eval data.")


def main() -> None:
    """Generate example data."""
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath
    sample_ratio = args.sample_ratio

    output_folderpath = os.path.dirname(output_filepath)

    input_data = pd.read_csv(input_filepath, sep='\t')

    with open(output_filepath, "wt") as fpw:
        input_data.drop_duplicates(keep='first', inplace=True)
        input_data = input_data.sample(frac=1.0, random_state=42)
        cut_idx = int(round(sample_ratio * input_data.shape[0]))
        if cut_idx < input_data.shape[0]:
            output_data, eval_data = input_data.iloc[:cut_idx], input_data.iloc[cut_idx:]
        else:
            output_data = input_data
            eval_data = pd.DataFrame()
        
        for row in output_data.itertuples():
            try:
                # make the length of the property fixed
                temp = format(getattr(row, 'y'), '0=+7.2f')
                fpw.write(
                    f"<rt>{temp}|{getattr(row, 'x')}{os.linesep}"
                )
            except:
                print(f"Problem processing {row}")
        if not eval_data.empty:
            with open(f'{output_folderpath}/eval.txt', "wt") as fpw2:
                for row in eval_data.itertuples():
                    try:
                        # make the length of the property fixed
                        temp = format(getattr(row, 'y'), '0=+7.2f')
                        fpw2.write(
                            f"<rt>{temp}|{getattr(row, 'x')}{os.linesep}"
                        )
                    except:
                        print(f"Problem processing {row}")


if __name__ == "__main__":
    main()
