# 1. Changes and updates

## 1.1. Multinomial logistic regression

Implemented such that in sequential forward selection, the performance metric used is accuracy, defined as
$$\frac{\text{number of correctly classified samples}}{\text{all samples}}$$
although this is flexible.


Mean AUC and individual AUCs are output as metrics for each model. In the runs discussed below, 4 classes were used, defined by separating the dataset into quartiles.

## 1.2. Thresholding

Author thresholding is not a new feature. An author threshold of 2 means that an author must have had 2 or more activities (comments or posts) on the subreddit during the (rolling) collection period - this is for ease of calculating metrics like mean author sentiment etc.


Thread size threshold - e.g. when looking at linear regression, we were only interested in the successful posts (a post that gained one comment or more), so previously, the data was filtered _before_ being processed by the class. _However_, due to the author thresholding mentioned above, this may have been causing needless data loss (in fact, it was very minimal), therefore it was added to the class so that both thresholding events could occur with minimal data loss.


## 1.3. Scaling and other considerations

- The model data is now scaled with scikit-learn.
- The quantile class division for multinomial regression occurs after author thresholding, as author thresholding could affect class size.
- Logging was implemented for better documentation of the regression runs.

## 1.4. Model convergence

- A function was introduced to handle models not converging, which changes the convergence algorithm and/or increases the number of iterations in order to ensure most (if not all) models converged.
- Model convergence method and number of iterations (where applicable) were added to the results spreadsheets.

## 1.5. Domains

Two extra model features were added, both based on the post domain:
- **domain pagerank**: the PageRank score of the domain, obtained from the downloadable csv on the [domcop website](https://www.domcop.com/top-10-million-websites),
- **domain count**: this is analogous to the author activity count - it's the number of times a domain has been used on the subreddit in the most recent collection period. Introduced because, although pagerank measures the most popular domains worldwide, certain subreddits gravitate towards more specific domains very frequently, or tend to self-reference.


# 2. Models and datasets

In all cases, the author activity threshold was set to 2, Forward Sequential Selection was used, and the same regressors were used.


The subreddits considered are r/Politics, r/Cryptocurrency, r/Books and r/Conspiracy.

## 2.1. Calibration and model periods
3 cases considered:
- (rolling) collection period 7 days, model period 7 days,
- (rolling) collection period 7 days, model period 14 days,
- (rolling) collection period 14 days, model period 7 days.

In all cases, the validation period was 7 days.

## 2.2. Regression types

- **Linear**: $R^2$ was used as the performance scoring method for the forward sequential selection, the thread size threshold was set to 2 (hence only threads that gained at least one comment were considered), and $y$ was thread size.

- **Logistic**: AUC was used as the performance scoring method for the FSS, and $y$ was success, equal to 1 if the post had 1 comment or more, 0 if the post had no comments.

- **Multinomial logistic**: Accuracy as defined above was used as the performance scoring method for the FSS. 4 classes were created using the 0.25, 0.5, and 0.75 quantiles. Thread size was thresholded such that only successful posts were considered, and $y$ was thread size.

## 2.3. Dataset sizes

# Graphs over the different collection and model periods
## 1.1. Linear regressions

In this case calibration and validation $r^2$ vs number of features are shown.