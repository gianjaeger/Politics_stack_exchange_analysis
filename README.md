# Text analysis of the "Politics" Stack Exchange

This project was completed as part of a take-home exam. It assessed the posts and comments made by the three most "polarizing" users on the "Politics" Stack Exchange forum. The exam, titled "Introduction to Data Science and Machine Learning in Python," consisted of three parts, and this project was completed as a response to the first part. It covers content taught by Bernie Hogan in the class "Fundamentals for Social Data Science in Python."

**The project approaches the problem as follows:**
1. Data on 100,000+ posts, comments and users is downloaded from the archive.
2. The three most polarizing users are identified as those with the largest number of combined UpVotes and DownVotes under the condition that the ratio between the two values lies between 2:3 and 3:2 or vice versa.
3. To determine if there is a meaningful trend in the topics covered by the most polarizing users, each post is grouped into one of three categories (current events, ideology or policy) with respect to their tags. A naive bayes classifier is then used to categorize posts with no tags based on their content.
4. k-means clustering is then used to derive trends in the content of the posts, which might have been neglected when using tags.
5. Their comments are then assessed via cosine similarity and compared to the rest of the corpus to analyse similarities and differences in the rethoric of the comments made by polarizing users.

--------

A PDF with the proposed methodology and the final answer - as submitted to the examiners - is also included in the repository.
