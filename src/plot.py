import matplotlib.pyplot as plt
import pandas as pd
import os

def plot(dataset, window, model, horizon, county):
    # y = pd.read_csv(f'result/{dataset}/{model}/{str(horizon)}/true.csv').values
    # y_hat = pd.read_csv(f'result/{dataset}/{model}/{str(horizon)}/pred.csv').values
    y = pd.read_csv(f'result/{dataset}/{window}/{model}/{str(horizon)}/true.csv').values
    y_hat = pd.read_csv(f'result/{dataset}/{window}/{model}/{str(horizon)}/pred.csv').values
    idx = COUNTY.index(county)
    if not os.path.exists(f'figures/{window}'):
        os.makedirs(f'figures/{window}')
    plt.figure(1)
    plt.plot(y[:,idx], label='Truth')
    plt.plot(y_hat[:,idx], label='Prediction')
    plt.legend()
    plt.title(f'{model} model {str(horizon)}th day prediction for {county}')
    plt.savefig(f'figures/{window}/{model}_{str(horizon)}_{county}.png')
    plt.close(1)

COUNTY = ['Alameda', 'Amador', 'Butte', 'Calaveras', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 
          'Glenn', 'Humboldt', 'Imperial', 'Kern', 'Kings', 'Lake', 'Los Angeles', 'Madera', 'Marin', 
          'Mendocino', 'Merced', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Riverside', 'Sacramento', 
          'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 
          'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 
          'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo', 'Yuba']

dataset = 'ca48-548'

# horizon = 7
# window = 7

# model = 'dummy'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'arma'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'colagnn'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'colagnn_noattn'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'colagnn_thresholding'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'colagnn_noattn_sci'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'colagnn_identityadj'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

# model = 'linear'
# plot(dataset, window, model, horizon, 'Los Angeles')
# plot(dataset, window, model, horizon, 'San Francisco')

def plot_multiple(dataset, window, models, horizon, county, out_filename):
    # y = pd.read_csv(f'result/{dataset}/{model}/{str(horizon)}/true.csv').values
    # y_hat = pd.read_csv(f'result/{dataset}/{model}/{str(horizon)}/pred.csv').values
    idx = COUNTY.index(county)
    y = pd.read_csv(f'result/{dataset}/{window}/{models[0]}/{str(horizon)}/true.csv').values
    plt.figure(1)
    for model in models:
        y_hat = pd.read_csv(f'result/{dataset}/{window}/{model}/{str(horizon)}/pred.csv').values
        plt.plot(y_hat[:,idx], label=model)
    plt.plot(y[:,idx], label='Truth')
    plt.legend()
    plt.title(f'{str(horizon)}th day prediction for {county}')
    plt.savefig(f'figures/{out_filename}.png')
    plt.close(1)
    
# models = ['dummy', 'arma', 'linear', 'colagnn', 'colagnn_noattn', 'colagnn_thresholding', 'colagnn_noattn_sci', 'colagnn_identityadj']

models = ['dummy', 'arma', 'colagnn', 'colagnn_noattn', 'colagnn_noattn_sci', 'colagnn_identityadj']

horizon = 7

plot_multiple(dataset, 7, models, horizon, 'Los Angeles', '7-7-Comparison-LA.png')
plot_multiple(dataset, 14, models, horizon, 'Los Angeles', '14-7-Comparison-LA.png')
plot_multiple(dataset, 28, models, horizon, 'Los Angeles', '28-7-Comparison-LA.png')

plot_multiple(dataset, 7, models, horizon, 'San Francisco', '7-7-Comparison-SF.png')
plot_multiple(dataset, 14, models, horizon, 'San Francisco', '14-7-Comparison-SF.png')
plot_multiple(dataset, 28, models, horizon, 'San Francisco', '28-7-Comparison-SF.png')
# model = 'dummy'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'arma'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'colagnn'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'colagnn_noattn'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'colagnn_thresholding'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'colagnn_noattn_sci'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'colagnn_identityadj'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')

# model = 'linear'
# plot(dataset, model, horizon, 'Los Angeles')
# plot(dataset, model, horizon, 'San Francisco')