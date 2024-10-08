# LohoRavens - Documentation


### Current Dataset Status:
- Manually written tasks: 23
- GenSim: 41
- Target: 50 tasks

### Build tasks (20th April):
- [x] Check tasks in the 'grounded decoding' (GD) paper. (12th April)
  - [x] Explore the possibility of adding objects besides blocks and bowls, such as letters, etc.
        **Not successful**
  - [x] Add box packing tasks.
  - [x] Reconstructing the GD tasks for our dataset (15 tasks added)
        
- [ ] Explore using GenSim to generate tasks (15th April)
  - [ ] We will try to increase the number of tasks to 50.

If we do not have enough tasks: **not needed**
- [ ] Consider adding tasks with fine-grained instructions, such as "stack blocks in the order of red, blue, green, and yellow.” (1 week)
- [ ] Add precise spatial reasoning tasks. (2 weeks)

In any case, we need to annotate the tasks:
- [ ] Adding some commonsense attributes, like cool/warm color to out dataset (incomplete)

### Task list
- [x] Color reasoning 
	- [x] Stack only the blocks of cool/warm/primary/secondary colors on a {color} zone.
	- [x] Stack only the blocks in alternate colors.
	- [x] Put the blocks in the bowls with matching colors.
	- [x] Put the blocks in the bowls with mismatching colors.
	- [x] Stack blocks of the same color.
- [ ] Size reasoning
	- [x] Stack blocks of the same size.
- [ ] Color + Size reasoning
	- [x] Stack only the bigger/smaller blocks of cool/warm/primary/secondary colors on the {color} zone.
	- [x] Stack only the blocks of cool/warm/primary/secondary colors in an ascending order from big to small on the {color} zone.
	- [x] Stack smaller blocks over bigger blocks of the same color.
	- [x] Stack blocks of the same color in the zone with the same color, with the bigger blocks underneath.
- [ ] Spatial reasoning
	- [ ] Absolute spatial
		- [x] Stack all the blocks on the {ABS_POS} area.
		- [x] Move all the blocks on the {ABS_POS} area to {ABS_POS} area.
		- [ ] {CORNER}
- [ ] Color + Spatial reasoning
	- Relative spatial
		- [x] Put all the blocks on the left/right/top/bottom of the red block on the {color} zone.
	- Absolute spatial
		- [x] Stack all the blocks on the {ABS_POS} area.
		- [x] Move all the blocks on the {ABS_POS} area to {ABS_POS} area.
		- [ ] {CORNER}
- [ ] Size + Spatial reasoning
	- Absolute spatial
		- [x] Stack all the bigger/smaller blocks on the {ABS_POS} area.
		- [x] Move all the bigger/smaller blocks on the {ABS_POS} area to {ABS_POS} area.
- [ ] Color + Size + Spatial reasoning
	- Relative spatial
		- [x] Put all the bigger/smaller blocks on the left/right/top/bottom of the {color} block on the {color} zone.
	- Absolute spatial
		- [x] Stack all the bigger/smaller {color} blocks on the {ABS_POS} area.
		- [x] Move all the bigger/smaller {color} blocks on the {ABS_POS} area to the {ABS_POS} area.
- [ ] Counting / Arithmetic reasoning
	- [x] Move all the blocks of a color that occur in even numbers to the same colored zone.
- [ ] Shape construction (check GenSim paper)
- [ ] Hidden object descriptions with distractors
	- [x] Put the blue block under the red block into the blue bowl. (There are two blue blocks on the table, where one is under the blue block, and the other one is visible.)
	- [x] Put the blue block on the first layer of the three-layer pyramid into the blue bowl. 


### Dataset check and classification
- [ ] Fix the evaluation metric bugs. (By April 19th)
- [ ] Check all the tasks and videos, especially for GenSim tasks. (By April 24th)
- [ ] Generate 200 instances as val/test sets for each task. (By April 25th)


### Baseline and analysis (End of Mai).
- [ ] Train a better CLIPort pick-and-place primitive.
- [ ] Find the reason of the poor performance of the primitive related to the absolute spatial reasoning. (May 3th)
- [ ] Add Code as Policy (CaP) as a baseline. (2 weeks, DDL: May 10th)
- [ ] Test different combinations of three LLMs and VLMs. (2 weeks)
- [ ] Analysis on reasoning capabilities. (2 weeks)

### Dataset Curation 
- [ ] Code release
- [ ] Validation/Test dataset release
- [ ] Task table publication


### Paper writing (June 5th, 9pm - NeurIPS)
- [ ] Initial draft (By April 29th)
