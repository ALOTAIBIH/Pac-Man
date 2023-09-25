# Pac-Man
### **Abstract**
	
For the Pac-Man competition, it was implemented distinct strategies for both offensive and defensive agents because they work individually based on the game state and circumstances they are working upon. In this wiki will describe the strategies and algorithms that used in designing the agent such as Monte Carlo Tree Search, (A*) Heuristic Search Algorithms. Besides tweaking the algorithm based on the game state, it was also applied some techniques to maximize the score while defending our areas of the food.
	
### **Introduction**

The Pac-Man game is a competition between the Red team and Blue team. Each team comprised of two agents -ghost for defensive and Pacman for attacking. The goal is to eat all the dots while avoiding the ghosts. In the beginning, we are observing and understanding the game environment and deciding what kind of agent might act. We examined the game by operating the baseline team and noted that establishing a set of features might be valuable in implementing a Mento-Carlo agent. We created different plans for both defensive and offensive agent because their actions on environments were different.

### **Techniques Used**
		
1. [Heuristic Search](Heuristic Agent)
2. [Monte Carlo Tree Search](Monte Carlo Agent)

### Challenges 

1. Locating the defensive agent to maximize the possibility of eliminating the opponent Pacman.

2. When the ghost is nearby, prioritizing the plan on how to eat the capsule 


3. Recognizing the suitable features and corresponding weights in Monto-Carlo.

4. Monitoring all the potential places of the next best move for our Pac-Man, if the opponent ghost is near in those places to evade getting killed. 

5. stuck the agent in the centre of the map and he is not able to restart the attack
	
### Strategies Which Worked
	
[Link to Strategies Page](https://github.com/COMP90054-classroom/contest-pac-pac-pac/wiki/Design-Choices)
1. Locating the ghost near the middle of the network improved in the quick discovery of the opponent Pacman.
2-Designing the Pacman move to either back home using A* to avoid getting killed or consume capsule if capsule available and it is not in the ghost way.
3. Tracing the location of the opponent based on the area of the food eaten.
4. Forcing our Pacman to back home and save food after calculating a percentage of total food available once it goes for the hunt. 
5. Dividing the main problem into sub-problems and using many plans to prioritize our strategy.

### **Evaluation Metric and Critical Analysis**	
	
[Link to Evaluation and Critical Analysis Page]
	
we assessed our agents versus baseline agents. Each other doing various random maps and examined their achievement to assist us to determine which one is doing great when presented to an unseen network.

### **Future Work**
		
1. using Game Theory to predict the opponent's moves.
2. Using the Q-learning algorithm due to ability navigate the map more efficiently, eating food is crucial to winning the game.
3. Making behaviour of our agents more autonomous.
	
### **Team Members**

1. Ankita Dhar – 1154197
2. Zexi Liu - 813212
3. Hissah Alotaibi- 1042537
	
### **Conclusion** 
	
In the beginning, we began by generating different strategy for both offensive and defensive agents. However, after rigorous analysis from running the replays, we recognised that it does not act appropriately. Therefore, we improved these strategies for our agents. We were dividing the problem into subproblems assisted us in planning better strategies to let agents more reliable in random layouts.  
