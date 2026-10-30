# Research Paper Retrieval Evaluation – MRKL Systems

### 1. Factual Questions

**Q1.** What was the learning rate used during the prompt-tuning experiments conducted with the J1-large model?  
Expected page: 8
Highest Scored page:- No. 14 with score = 0.7239
Expected page score:- 0.6693 (ranked 3rd)

**Q2.** What was the name of the AI21 Labs implementation of the MRKL architecture?  
Expected page: 4
Highest Scored page:- No. 1 with score= 0.7062
Expected page score:- 0.7052 (ranked 2nd)

**Q3.** How many prompt tokens were used during the prompt-tuning experiments?  
Expected page: 8
Highest Scored page:- No. 14 with score= 0.7693
Expected page score:- 0.7376 (ranked 2nd)

### 2. Methodology Questions

**Q4.** How does the MRKL architecture route incoming natural language inputs to different expert modules, and what happens when no suitable expert is available?  
Expected page: 4 
Highest Scored page:- No. 4 with score 0.8521 (Got the expected page!!!!)

**Q5.** How does the data augmentation methodology generate training and testing examples for arithmetic operations, and what dimensions characterize each example?  
Expected page: 7–8
Highest Scored page:- No. 7 with score 0.883 (Got the expected page!!!)

**Q6.** How does Jurassic-X use a language model and a symbolic calculator together to solve arithmetic problems expressed in natural language?  
Expected page: 5–7
Highest Scored page:- No. 7 with score 0.8085 followed by page 5 with score 0.7735 (Got the expected pages!!!)

### 3. Comparison and Results Questions

**Q7.** How did Jurassic-X's arithmetic generalization performance compare with GPT-3 when tested on numbers containing different numbers of digits?  
Expected page: 9
Highest Scored page:- No. 11 with score= 0.8108
Expected page score:- 0.7832 (ranked 3rd)


**Q8.** How did prompt-tuning compare with few-shot prompting when generalizing to arithmetic questions where operands were expressed as words instead of digits?  
Expected page: 14
Highest scored page:- No. 14 with a score 0.8890 (highest score till now and got the expected page!!)

### 4. Limitations Question

**Q9.** What limitations or failure cases did the authors identify when generalizing from single-operation arithmetic problems to two-operation problems?  
Expected page: 12–13
Highest Scored page:- No. 11 with score= 0.7954
Expected page score:- NOT IN TOP K (Very weird but i guess this happened bcz the limitations werent condensed in any pages instead they were spread around the whole paper)

### 5. Unanswerable Question

**Q10.** What was the exact computational cost (GPU hours and monetary cost) of training the Jurassic-X arithmetic extraction model?  
Expected page: Not specified in the paper (Unanswerable)
Highest scored page:- Page 11 with score 0.773

### 5. Totally Unrelated question

**Q11.** What is the capital of germany?
Expected: Not answerable
HIghest scored page:- No. 1 with score of 0.413