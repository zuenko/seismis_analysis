# Seismic Interpretation analysis with deeplarning

## Datasets
### Ours
For Unet-3d we propose our labled dataset in masks+slices folder. It consist of images sliced horizontally from SEG-Y files from New Zeland.

### Rosneft booster's challenge

[Link](https://boosters.pro/championship/seismic_challenge/) Labled crosslines split for seismic facies analysis. Used

### Netherlands Dataset

[Link](https://arxiv.org/pdf/1904.00770.pdf) Contains pos-stackdata, 8 horizons and well logs of 4 wells. For the purposes of ourmachine  learning  tasks,  the  original  dataset  was  reinterpreted,generating 9 horizons separating different seismic facies intervals.The interpreted horizons were used to generate∼190,000 labeledimages  for  inlines  and  crosslines. 

## Results

### Crosslines

Our Unet + Resnet model achived 0.97 (max value - 1.00) multidice metric on the validation set.