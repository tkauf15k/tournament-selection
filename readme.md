# Tournament Selection

Input: number of rounds, number of players and skill specs of each player

Generates a match table for each round, considering constraints for jokers & mixed/unmixed
Samples pairs according to some constraints randomly, then validates the feasibility based on *perfect matching*.
Finally, takes a random solution from the top *n* fittest ones.

Decrease *num_samples* to decrease execution time.
Increase *num_top* to allow for more unbalanced matches. 

## Todo
- [ ] implement penalization to prefer different pairs being selected