# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Data for the model was provided by Barry Becker who extracted the data from a 1994 Census database. Sickit-learn's RandomForestClassifier was used to train and develop this model.

## Intended Use
The intended use is to predict whether an individual has an incomve greater than or less than $50,000 based on the available factors in the census data.

## Training Data
The training data is extracted from a 1994 census database performed by Barry Becker. It contains 32,562 rows with the following 15 columns.

- age
- workclass
- fnlgt
- education
- educatoin-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country
- salary

80% of data was randomly selected to train the model.

## Evaluation Data
Data was evaluated using the remaining 20% of data that was not used during training.

## Metrics
Metrics used to evaluate the model as well as its results are listed below.

- Precision: 0.7502
- Recall: 0.6231
- F1: 0.6808

## Ethical Considerations
This model should be used for research purposes and should not be considered a method of evaluating an individual's worth. The method of collecting the data into the census database is unclear and could include inaccuracries and or a lack of representation for specific demographics which could lead to biases in the results. 

## Caveats and Recommendations
For future training of this model, a more recent census dataset should be used, as significant changes in work culture have occurred over the past 30 years. These include a greater emphasis on gender and ethnic diversity, evolving educational requirements, and shifts in the average age of workers. Additionally, household structures have transformed, with changes in marital status, primary earners, and birth rates.
