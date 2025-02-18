# Parameters and Structure

The present model has been based on the cossine similarity calculation, which means that we rely on a more simple approach trying to figure similar content to recommend to our users.
Using this approach our **'ContentRecomender'** class uses as main parameter a top_k value.
The top_k has the function to determine how many similar content per base news are returned for a user.
* For example: User 'A' has 5 news of reference and top_k is defined in 10, in such scenario we will recomend the top 10 similar news for each reference news resulting in a json with 50 recommendations which can then be further filtered on the application side (for example bringing only top 15 to the front end page)

Our model class has also been documented so every method has been described in its function with input/output parameters. 