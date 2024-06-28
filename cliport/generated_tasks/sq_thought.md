### Comments from sq on May 1st
- [ ] A general problem of evaluation for the tasks which have multiple correct answers: 
take the `ArangeBlocksBasedOnColorAndSize` as an example, there are many correct final poses, 
however, the current `matches` in the code defines only one correct pose for every block with `np.eye`. 
What I mean is that the smaller yellow block can be at any four positions on the left side in the 
video, but as 
defined in the current code, there is only one correct pose for the smaller yellow block. 
I think this problem exists in many tasks which include multiple correct answers.
Could you try replacing `np.eye` with `np.ones` for the `matches` matrix to fix this problem?
You could check one example code for the task `StackBlocksByColor`.
- [ ] Code format: We had better make the code format style consistent. 
You could try to use the reformatting tool in the IDE to reformat the code.
My habit is to limit the maximum line length to 99. 
Can you reformat the code from your side like this, too?
- [ ] Other comments starting with `SQ:` below.




### General:
- [ ] Can we remove the wooden pallet?

--**removing pallet is possible**

- [ ] Which tasks are the same as the tasks in GenSim?

--**build car**


### Tasks:
- [ ] ArrangeSameColorShapes: why do we need to move the  red block and the blue block even if they are already on the left or the right, respectively.

--**edited**

- [ ] arrange blocks based on color and size: based on a specific pattern? What is the specific pattern?

--**horizontally size, verticaly color**

- [ ] assemble word: 
    - [ ] from the video, the orientations of the letters are not consistent. In this way, they don't look good. Can we make their orientation consistent?

      --**according to task code oracle should care about orientation, I don't figure out why it performs bad when dealing with kitting assets**

    - [ ] Why does the robot manipulate one letter many times?

      --**when oracle think obj and target don't match it manipulate obj again and again until it think they match**

- [ ] Build car: from the video, it's hard to say the final shape is a car.

--**to stand a cylinder turns out to be difficult so the wheels lie down**

- [ ] Build house: the video is not successful.

--**new uploaded**

- [ ] Build rectangular pyramid: some blocks seem redundant as seen in the video.

--**not redundant, some blocks' initial poses occupy target poses, and oracle doesn't know to clear up the target position first, this is also a major problem of ravens environment**

**SQ: but can we choose a free area first for building the pyramid? 
Another problem is the evaluation. Because building the pyramid at any position on the table 
can be the correct answer. So the solution may be to limit the target position with a 
colored zone (e.g., build the pyramid on the red zone.) unless you can find a way to define the 
correct match matrix.
This problem may also exist in other tasks such as construct circle with blocks. 
Could you check this please?**

- [ ] Color coordinated box pyramid: 
    - [ ] It's not like a pyramid?

       --**GPT named it a pyramid**

    - [ ] What's the instruction "stack the {color} box on top of the {prev_color} box." meaning? It is not like a high-level instruction.

       --**this instruction is GPT generated, I now corrected it to be relatively high-level**

- [ ] Color matching block in pallet: I don't think it's a meaningful task and can be removed.

--**this task means to place blocks on corners of the pallet**

- [ ] Color matching blocks to corner and zone: can be removed.

--**this task means to place blocks on corresponding position anchors first and then place in corresponding zones**

- [ ] Ball: should we include balls in our benchmark? Except the shape, do they have difference from blocks in terms of reasoning abilities?

--**balls are not stable since they tends to roll out of the table so that oracle cannot see them and will skip the step, oracle lacks a function to stop the rolling**

- [ ] Cylinder: same as ball.

--**when cylinders lie down they function the same as blocks, and as mentioned above I didn't manage to stand cylinders up so cylinders won't roll. If their initial poses are set standing up they stuck in the desktop and cannot be moved. I don't know why**

- [ ] Construct circle with blocks: from the video, some blocks are outside the table.

--**edited by resetting circle center and radius.**

- [ ] Container stack puzzle: similar to existing tasks and can be removed.
- [ ] Insert block to different color fixture: can we ensure every block is exactly in the fixture? Because from the video, I see some blocks are not placed well in the fixture.

--**maybe make fixtures larger?**

- [ ] Rotate and insert cylinder: the demonstration fails in the video.

--**as mentioned above oracle cannot find a good way to stand cylinders up**

- [ ] Multi level cylinder tower: remove.

--**this taks means to stack cylinders and block one layer by one layer**

- [ ] Precision cylinder tower: leave it considered. We should think whether we should indeed include precise manipulation tasks or not.
- [ ] Put piles into zone: similar to the task in CLIPort. And the demonstration fails according to the video.

--**from the second last step there was one blue block in red zone so oracle tried to push it into blue zone, but the spatula is too big so some other red blocks were involved, if self.max_steps is larger oracle will keep pushing**

**SQ: but anyway, we should ensure the created demonstrations are correct, right? so we should find a way to check the final poses and remove the wrong demonstrations.**

- [ ] Put block into bowl with different color: remove since it's same as the previous.
- [ ] Put even number blocks into same color bowl: how many blocks should be put into bowl? Let's say, two or four?

--**oracle will choose a random even number smaller then the number of blocks**

**SQ: for the current code, you specify the number. In this way, it's an easy task which does 
not involve number reasoning. Can we think about another way to modify the task?**

- [ ] Put odd number blocks into same colored zone: demonstration fails.

--**demo doesn't fail as there are odd numbers of blocks in each corresponding zone**

- [ ] Sequential block insertion and stacking: the zones are underneath the pallet. What's the meaning of "insertion into the zone"?

--**from GPT's description and code the task means to first place blocks on pallet meanwhiles exactly above the corresponding zone then stack the blocks on pallet to form a 2-d pyramid**

- [ ] Sequential hurdle course: interesting, but demonstration fails.

--**the push task needs some luck, it is an environment problem**

- [ ] Sort shapes in zones: how can blocks have different shapes? I'm confused about this setting. In the video, there is only one shape.

--**this is a wrong task. GPT means to sort different shapes of block into different shapes of zones, however there is no asset of other shape of blocks nor zones**

- [ ] Split piles into zone: I cannot get the meaning.

--**this task means to split the pile into two halves and let oracle push each half into a different zone. The problem is that the blocks in pile have orders, so oracle will try to push specific blocks into specific zone, and due to spatula size other irrelative blocks might get involved. I don't know how to edit the push function to let oracle ignore the block order**

- [ ] Put blocks on zone bisector: why are there two overlap zones.

--**the zones are generated with random initial poses, overlapping is normal**
