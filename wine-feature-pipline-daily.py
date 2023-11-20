import os 
import modal
import pandas as pd
import numpy as np
from datetime import datetime

import os

LOCAL = False

import os
os.environ['HOPSWORKS_API_KEY'] = 'rd3AnewLiGn44iYx.ECMxeKjZYC48N5Qm4BFYWCH5oWrAAamQ4GEdJ3D03l6hjlFlLjG3pApQXK9MqJml'


from datetime import datetime 
import numpy as np
import pandas as pd

def calculate_stats_by_quality(wine_df):
    quality_stats = {}
    numeric_cols = wine_df.columns.drop(['date_added', 'id', 'type', 'quality'])

    print(numeric_cols)
    # Iterate through each unique quality value
    for quality in wine_df['quality'].unique():
        filtered_df = wine_df[wine_df['quality'] == quality]

        filtered_df = filtered_df[numeric_cols]
        
        # Calculate mean and standard deviation for the filtered data
        mean = filtered_df.mean()
        std = filtered_df.std()

        # Store the stats in the dictionary
        quality_stats[quality] = {"mean": mean, "std": std}

    return quality_stats

def generate_random_datapoint(stats_by_quality, num_datapoint):
    # Randomly select a quality from 3 to 9
    quality = np.random.randint(3, 10)

    # Retrieve the stats for the selected quality
    stats = stats_by_quality[quality]

    # Generate a random datapoint using the mean and std for the selected quality
    data = pd.DataFrame()

    for _ in range(num_datapoint):
        random_datapoint = {feature: np.random.normal(mean, std) 
                            for feature, (mean, std) in zip(stats['mean'].index, zip(stats['mean'], stats['std']))}

        # Assuming 'type' needs to be cast to an integer
        if 'type' in random_datapoint:
            random_datapoint['type'] = int(random_datapoint['type'])

        # Randomly select a quality from 3 to 9 (or adjust as needed)
        quality = np.random.randint(3, 10)
        random_datapoint["quality"] = quality
        type = np.random.randint(0,2)
        random_datapoint["type"] = type
        
        # Assuming wine_df is your existing DataFrame

        # Add a new column 'date_added' with today's date
        random_datapoint["date_added"] = datetime.today().date()

        # Convert to DataFrame
        random_datapoint_df = pd.DataFrame([random_datapoint])

        # Append to data
        data = pd.concat([data, random_datapoint_df], ignore_index=True)

    return data

if LOCAL == False:

    stub = modal.Stub('wine_insert')

    image = modal.Image.debian_slim().pip_install(["hopsworks", "numpy", "pandas", "python-dotenv"])

    @stub.function(image = image, schedule = modal.Period(days = 1), secret = modal.Secret.from_name('hopsworks_api_key'))

    def f():
        import hopsworks

        project = hopsworks.login(api_key_value= os.environ['HOPSWORKS_API_KEY'])
    
        fs = project.get_feature_store()         

        fg = fs.get_feature_group(name = 'wine_v2', version = 2)
        wine_df = fg.read(read_options={"use_hive": True})

        if 'id' in wine_df.columns and not wine_df['id'].empty:
            start_id = wine_df['id'].max() + 1
        else:
            start_id = 0


        stats = calculate_stats_by_quality(wine_df)
        datapoints = generate_random_datapoint(stats, 9)

        datapoints['id'] = range(start_id, start_id + len(datapoints))

        fg.insert(datapoints)


if __name__ == '__main__':
    if LOCAL == True:
        f.local()
    else:
        modal.runner.deploy_stub(stub, 'wine')
        with stub.run():
            f.remote()


    

    
