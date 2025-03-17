import pandas as pd

def compare_csv():
    # Load the CSV files
    df1 = pd.read_csv('data/sanity/output/actions.csv')
    df2 = pd.read_csv('data/sanity/output/pred_actions.csv')

    # Merge on 'video_name', 'start_frame', and 'end_frame'
    merged_df = pd.merge(df1, df2, on=["video_name", "start_frame", "end_frame"])
    
    # Ensure index alignment by resetting index
    merged_df = merged_df.reset_index(drop=True)

    # Extract "action_id" and "pred_action_id" columns
    actions = merged_df['action_id']
    pred_actions = merged_df['pred_action_id']
    
    # Compute accuracy across "action" and "pred_action"
    accuracy = (actions == pred_actions).mean()
    return accuracy

if __name__ == '__main__':
    accuracy = compare_csv()
    print(f'Accuracy: {accuracy:.2%}')    