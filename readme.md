# Tournament Selection

Input: number of rounds, number of players and skill specs of each player

Generates a match table for each round, considering constraints for jokers & mixed/unmixed
Samples pairs according to some constraints randomly, then validates the feasibility based on *perfect matching*.
Finally, takes a random solution from the top *n* fittest ones.

Decrease *num_samples* to decrease execution time.
Increase *num_top* to allow for more unbalanced matches. 

## Todo
- [ ] implement penalization to prefer different pairs being selected


## Run

```bash

pip install -r requirements.txt

python tournament.py --num-players 11 --num-rounds 10 

```