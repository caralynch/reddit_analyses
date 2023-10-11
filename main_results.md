# To do
- run linear regressions & produce graphs for each subreddit, with:
    - num comments
    - num participating authors
    - post score
vs R2 in each graph. Do thresholded/not thresholded, with standard collection window of 1 wk + max mod period, with 1 wk validation (could also try 1 day, 3 days validation, see if changes). Discuss low R2. incl stdev

- Check when author_activity_calibrate_validate_regressions_v2 was created - why AUC when should be looking at R2? Was this an early draft??

- Logistic regression: feature selection max days v5 looks to be best file, check this was the latest. The graphs are a bit shit - check which graphs should actually work?? Does logistic regression produce some kind of fit? I think it does. If so graph said fit. Tables are more useful for what i've got atm, include pvals etc. v4 is proper feat selection - I think v5 i gave it the top feats. incl stdev in graphs

- make some sort of table of top 5 features of each subreddit?? need to figure out smart way of displaying this. I feel like we did it with G but idk where spreadsheet is??

- dataset_summaries has a lot of the data i'd want to display about the subreddits, although summaries of r/pol are harder to find - might want to rerun some of these to incl r/politics
    
## New analyses
- post domains in logistic and linear regressions?
- linear regression when restricting to threads under/over a certain size?


# Main Results

## 0. Different subreddits have different surface-level characteristics

### Qualitative information
- specific rules to each subreddit

### Users
- Total number of users (in whole dataset, weekly, daily)
- Average number of posts per user (in particular users who only post once)
- Any user overlap between subreddits?
- Average sentiment score of users (can look at this thresholded/not thresholded)

### Threads
- Number of threads _after cleaning_ (in whole dataset, average weekly, daily)
- Number of threads that don't start (in whole dataset, average weekly, daily)
- In threads that start, average number of comments, average depth, median etc
- Average sentiment score of threads, median, mean etc

### Network views
- If links between subreddits, show each subreddit as a node and thickness of link propto nb of users that overlap?

## 1. Predicting whether a thread will start: logistic regressions

Non-negligeable portion of threads never start.

- logreg AUCs/graphs for each subreddit:
    - different data collection periods? different calibration periods? e.g. 1 week, 3 days, 1 day
    - what author thresholds used?
- emphasis on main characteristics not post characteristics
- subreddits exhibit similarities in feats which lead to thread success

## 2. Poor performance for prediction of size of threads that start: linear regressions
- both in terms of number of commens and of participating authors

## 3. Classifying threads as in Alvarez et al.
- Can fut truncated power laws to CCDFs of (thread sizes???)
- differently classified threads have different tail exponents in all subreddits
- neutral threads are always highest ranked (so last the longest) **across all subreddits**

