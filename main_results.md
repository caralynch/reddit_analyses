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

# To do
- post domains in logistic and linear regressions?
- linear regression when restricting to threads under/over a certain size?