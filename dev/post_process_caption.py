import pandas as pd
import re

def reformat_caption_csv(csv_path):

    df = pd.read_csv(csv_path)

    # Clean up the caption column
    df['caption'] = df['caption'].map(
        lambda x: re.sub(r' {2,}', '\\\\s', x.replace('\n', '\\\\n').strip())
        )

    csv_path_copy = csv_path.replace('.csv', '_copy.csv')
    df.to_csv(csv_path_copy, index=False)

if __name__ == "__main__":
    reformat_caption_csv('results/data/captions/vllama3.csv')